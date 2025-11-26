from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from bot.core.config import load_config
from bot.exchange_info import ExchangeInfoManager, SymbolFilters
from bot.execution.balance_manager import BalanceManager
from bot.execution.exchange_client import ExchangeRequestError
from bot.execution.live import LiveTrader
from bot.execution.market_loop import EntryPlan, ExitPlan, MarketLoop
from bot.execution.order_manager import OrderManager, OrderRequest, OrderPlacementError
from bot.execution.risk_gate import RiskCheck, RiskGate
from bot.execution.runners import MarketContext, PaperPosition
from bot.risk import ExternalSignalGate, MultiTimeframeFilter, PositionSizer, RiskEngine
from bot.risk.sizing import SizingContext
from bot.signals.strategies import StrategyParameters, StrategySignal


class FakeExchangeClient:
    def __init__(self, price: float = 100.0) -> None:
        self.price = price
        self.orders: list[Dict[str, Any]] = []
        self.position_payload: list[Dict[str, Any]] = []
        self.balance_payload: list[Dict[str, Any]] = [
            {
                "asset": "USDT",
                "balance": "1000",
                "availableBalance": "900",
            }
        ]
        self.pending_error: Optional[ExchangeRequestError] = None

    def set_balance(self, total: float, available: float) -> None:
        self.balance_payload = [
            {
                "asset": "USDT",
                "balance": str(total),
                "availableBalance": str(available),
            }
        ]

    def set_positions(self, payload: list[Dict[str, Any]]) -> None:
        self.position_payload = payload

    def change_leverage(self, **_: Any) -> Dict[str, Any]:  # pragma: no cover - interface compat
        return {"leverage": 10}

    def place_order(self, **payload: Any) -> Dict[str, Any]:
        if self.pending_error:
            exc = self.pending_error
            self.pending_error = None
            raise exc
        order_id = len(self.orders) + 1
        symbol = payload.get("symbol", "BTCUSDT")
        side = payload.get("side", "BUY")
        reduce_only = payload.get("reduceOnly") == "true"
        quantity = float(payload.get("quantity", 0.0))
        direction = 1 if side == "BUY" else -1
        if reduce_only:
            self.position_payload = []
        elif quantity > 0:
            self.position_payload = [
                {
                    "symbol": symbol,
                    "positionAmt": str(quantity * direction),
                    "entryPrice": str(self.price),
                    "markPrice": str(self.price),
                    "unRealizedProfit": "0",
                    "leverage": "5",
                    "positionSide": "BOTH",
                }
            ]
        response = {
            "orderId": order_id,
            "clientOrderId": payload.get("newClientOrderId", f"fake-{order_id}"),
            "executedQty": payload.get("quantity", "0"),
            "avgPrice": payload.get("price", str(self.price)),
            "price": payload.get("price", str(self.price)),
        }
        self.orders.append({**payload, **response})
        return response

    def get_balance(self) -> list[Dict[str, Any]]:
        return self.balance_payload

    def get_position_risk(self) -> list[Dict[str, Any]]:
        return self.position_payload

    def get_account_trades(self, **_: Any) -> list[Dict[str, Any]]:
        return []


class StubLogger:
    def __init__(self) -> None:
        self.records: list[Dict[str, Any]] = []

    def log(self, payload: Dict[str, Any]) -> None:
        self.records.append(payload)


@dataclass
class DummyBalanceManager:
    available_balance: float = 1_000.0

    def resolve_reduce_only_quantity(self, symbol: str, side: str, requested: float, price: float | None, filters: SymbolFilters | None) -> float:
        return requested


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def filters() -> SymbolFilters:
    return SymbolFilters(min_qty=0.001, min_notional=5.0, step_size=0.001, tick_size=0.1, max_leverage=20.0)


@pytest.fixture()
def exchange_info(cfg, filters):
    return ExchangeInfoManager(cfg, client=None, prefetched={"BTCUSDT": filters})


def make_params(cfg) -> StrategyParameters:
    defaults = cfg.strategy.default_parameters
    return StrategyParameters(
        fast_ema=int(defaults["fast_ema"]),
        slow_ema=int(defaults["slow_ema"]),
        rsi_length=int(defaults["rsi_length"]),
        rsi_overbought=float(defaults["rsi_overbought"]),
        rsi_oversold=float(defaults["rsi_oversold"]),
        atr_period=int(defaults["atr_period"]),
        atr_stop=float(defaults["atr_stop"]),
        atr_target=float(defaults["atr_target"]),
        cooldown_bars=int(defaults["cooldown_bars"]),
        hold_bars=int(defaults["hold_bars"]),
    )


def test_order_manager_generates_id_and_caches(cfg):
    fake_client = FakeExchangeClient()
    stub_logger = StubLogger()
    manager = OrderManager(cfg, client=fake_client, logger=stub_logger, balance_manager=DummyBalanceManager(), clock=lambda: 120.0)
    request = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=0.01)
    first = manager.submit_order(request)
    assert first.client_order_id.startswith("BTCUSDT-BUY")
    duplicate = manager.submit_order(request)
    assert duplicate.duplicate is True
    assert len(fake_client.orders) == 1


def test_order_manager_raises_structured_errors(cfg):
    fake_client = FakeExchangeClient()
    fake_client.pending_error = ExchangeRequestError("new_order", "auth", "denied", code=-2015)
    manager = OrderManager(cfg, client=fake_client, logger=StubLogger(), balance_manager=DummyBalanceManager(), clock=lambda: 10.0)
    request = OrderRequest(symbol="BTCUSDT", side="BUY", quantity=0.01)
    with pytest.raises(OrderPlacementError) as excinfo:
        manager.submit_order(request)
    assert excinfo.value.error.category == "auth"


def test_balance_manager_tracks_positions(cfg, exchange_info, filters):
    fake_client = FakeExchangeClient()
    fake_client.set_balance(total=1_200, available=1_000)
    fake_client.set_positions(
        [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.02",
                "entryPrice": "100",
                "markPrice": "101",
                "unRealizedProfit": "2",
                "leverage": "5",
                "positionSide": "BOTH",
            }
        ]
    )
    risk_engine = RiskEngine(cfg)
    manager = BalanceManager(cfg, fake_client, risk_engine=risk_engine, exchange_info=exchange_info, clock=lambda: 0.0)
    snapshot = manager.refresh_account_balance(force=True)
    assert snapshot.available == pytest.approx(1_000)
    positions = manager.sync_positions(force=True)
    assert positions is not None and "BTCUSDT" in positions
    assert manager.live_position_quantity("BTCUSDT", 1) == pytest.approx(0.02)
    assert manager.estimate_equity() >= snapshot.total


def test_risk_gate_blocks_invalid_orders(cfg):
    engine = RiskEngine(cfg)
    gate = RiskGate(cfg, engine)
    exposure_state = engine.compute_exposure({})
    decision = gate.assess_entry(
        RiskCheck(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.0,
            price=100.0,
            available_balance=1_000.0,
            equity=1_000.0,
            exposure=exposure_state,
            filters=None,
        )
    )
    assert decision.allowed is False
    engine._flatten_positions = True  # type: ignore[attr-defined]
    flatten_decision = gate.assess_entry(
        RiskCheck(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.01,
            price=100.0,
            available_balance=1_000.0,
            equity=1_000.0,
            exposure=exposure_state,
            filters=None,
        )
    )
    assert flatten_decision.should_flatten is True


def test_market_loop_generates_entry_and_exit(cfg, exchange_info, filters):
    params = make_params(cfg)
    ctx = MarketContext(symbol="BTCUSDT", timeframe="1m", params=params)
    signal = StrategySignal(index=1, direction=1, entry_price=102.0, stop_loss=100.0, take_profit=105.0)
    ctx.strategy.generate_signals = lambda frame: [signal]
    sizer = PositionSizer(cfg)
    multi_filter = MultiTimeframeFilter(cfg)
    external_gate = ExternalSignalGate(cfg)
    risk_engine = RiskEngine(cfg)
    risk_gate = RiskGate(cfg, risk_engine)
    balance = 1_000.0
    equity = 1_000.0
    available_balance = 1_000.0

    def sizing_builder(local_ctx: MarketContext, price: float, volatility: Dict[str, float]) -> SizingContext:
        return SizingContext(
            symbol=local_ctx.symbol,
            balance=balance,
            equity=equity,
            available_balance=available_balance,
            price=price,
            params=local_ctx.params,
            volatility=volatility,
            filters=filters,
            max_notional=None,
            symbol_exposure=0.0,
            total_exposure=0.0,
            active_symbols=0,
            symbol_already_active=False,
        )

    loop = MarketLoop(
        ctx,
        cfg,
        risk_gate=risk_gate,
        sizer=sizer,
        multi_filter=multi_filter,
        external_gate=external_gate,
        exchange_info=exchange_info,
        sizing_builder=sizing_builder,
        timestamp_fn=lambda row: "2024-01-01T00:00:00Z",
        log_sizing_skip=lambda *_: None,
    )
    frame = pd.DataFrame(
        [
            {"close": 100, "high": 101, "low": 99},
            {"close": 102, "high": 103, "low": 101},
        ]
    )
    latest = frame.iloc[-1]
    exposure = risk_engine.compute_exposure({})
    entry_plan = loop.plan_entry(
        frame,
        latest,
        balance=balance,
        equity=equity,
        available_balance=available_balance,
        exposure=exposure,
    )
    assert entry_plan is not None
    exit_plan = loop.plan_exit(entry_plan.position, pd.Series({"high": 106, "low": 99}))
    assert isinstance(exit_plan, ExitPlan)


class DummyLearningStore:
    def best_params(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return None

    def trade_count(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def record_trade(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - no-op stub
        return None


def test_live_trader_smoke_runs_single_step(cfg, exchange_info, filters):
    fake_client = FakeExchangeClient()
    fake_client.set_positions([])
    params = make_params(cfg)
    markets = [("BTCUSDT", "1m", params)]
    trader = LiveTrader(
        markets,
        exchange_info,
        cfg,
        client=fake_client,
        learning_store=DummyLearningStore(),
    )
    stub_logger = StubLogger()
    trader.order_logger = stub_logger
    trader.trade_logger = types.SimpleNamespace(log=lambda payload: None)
    trader.order_manager._logger = stub_logger

    def fake_frame(_ctx: MarketContext, _lookback: int):
        return pd.DataFrame(
            [
                {"close": 100, "high": 101, "low": 99},
                {"close": 102, "high": 103, "low": 101},
            ]
        )

    trader._fetch_frame = types.MethodType(lambda self, ctx, lookback: fake_frame(ctx, lookback), trader)

    entry_position = PaperPosition(
        symbol="BTCUSDT",
        timeframe="1m",
        direction=1,
        quantity=0.01,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=104.0,
        opened_at="2024-01-01T00:00:00Z",
    )
    entry_signal = StrategySignal(index=0, direction=1, entry_price=100.0, stop_loss=99.0, take_profit=104.0)
    entry_plan = EntryPlan(
        position=entry_position,
        signal=entry_signal,
        order_request=OrderRequest(symbol="BTCUSDT", side="BUY", quantity=0.01),
    )
    exit_plan = ExitPlan(side="SELL", quantity=0.01, price=104.0, reason="target")

    class StubLoop:
        def __init__(self) -> None:
            self._entered = False
            self._closed = False

        def plan_entry(self, frame, latest_row, **kwargs):
            if self._entered:
                return None
            self._entered = True
            return entry_plan

        def plan_exit(self, position, row):
            if self._entered and not self._closed:
                self._closed = True
                return exit_plan
            return None

    async def run_once() -> None:
        trader._market_loops["BTCUSDT"] = StubLoop()
        await trader._before_loop()
        await trader._before_step()
        trader._step_all()

    asyncio.run(run_once())
    assert fake_client.orders  # entry+exit recorded
