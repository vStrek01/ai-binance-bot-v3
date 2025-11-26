from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import pytest

from backtest.runner import BacktestRunner
from bot.backtester import Backtester
from bot.core.config import ensure_directories, load_config
from bot.exchange_info import ExchangeInfoManager, SymbolFilters
from bot.strategies import build_parameters
from core.models import Candle, Position, RiskConfig, Side, Signal
from core.risk import RiskManager
from core.state import PositionManager

TEST_SYMBOL = "TESTUSDT"
TEST_INTERVAL = "1m"


class PulseLongStrategy:
    """Deterministic strategy that opens then flattens every bar for predictable PnL."""

    strategy_mode = "pulse"

    def evaluate(self, market_state):  # pragma: no cover - exercised via integrations
        if market_state.open_positions:
            return Signal(action=Side.FLAT, confidence=1.0, reason="exit")
        return Signal(action=Side.LONG, confidence=1.0, reason="enter", stop_loss_pct=0.01, take_profit_pct=0.02)


def _risk_config() -> RiskConfig:
    return RiskConfig(
        max_risk_per_trade_pct=0.001,
        max_daily_drawdown_pct=0.99,
        max_open_positions=1,
        max_leverage=5,
        taker_fee_rate=0.0,
        maker_fee_rate=0.0,
        slippage=0.0,
        max_symbol_notional_usd=1_000_000.0,
        min_order_notional_usd=1.0,
    )


def _build_runner(strategy: PulseLongStrategy | None = None) -> BacktestRunner:
    strat = strategy or PulseLongStrategy()
    risk_manager = RiskManager(_risk_config())
    position_manager = PositionManager(equity=10_000.0, run_mode="backtest")
    return BacktestRunner(strat, risk_manager, position_manager, initial_equity=10_000.0, spread=0.0)


def _synthetic_candles(pattern: str, count: int = 60) -> List[Candle]:
    now = datetime.utcnow()
    price = 150.0
    candles: List[Candle] = []
    for idx in range(count):
        if pattern == "up":
            delta = 1.5
        elif pattern == "down":
            delta = -1.5
        else:  # chop
            delta = 0.75 if idx % 2 == 0 else -0.75
        open_time = now + timedelta(minutes=idx)
        close_time = open_time + timedelta(minutes=1)
        open_price = price
        price = max(25.0, price + delta)
        close_price = price
        high = max(open_price, close_price) + 0.4
        low = min(open_price, close_price) - 0.4
        candles.append(
            Candle(
                symbol=TEST_SYMBOL,
                open_time=open_time,
                close_time=close_time,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=1_000.0,
            )
        )
    return candles


@pytest.mark.parametrize(
    "pattern,expectation",
    [
        ("up", ">"),
        ("down", "<="),
    ],
)
def test_golden_trend_baselines(pattern: str, expectation: str) -> None:
    runner = _build_runner()
    result = runner.run(_synthetic_candles(pattern))
    pnl = result["summary"]["net_pnl"]
    if expectation == ">":
        assert pnl > 0, f"expected uptrend PnL to be positive, got {pnl}"
    else:
        assert pnl <= 0, f"expected downtrend PnL <= 0, got {pnl}"


def test_chop_environment_has_controlled_drawdown() -> None:
    runner = _build_runner()
    result = runner.run(_synthetic_candles("chop"))
    summary = result["summary"]
    assert abs(summary["net_pnl"]) < 200.0  # <2% drift on 10k equity
    assert summary["max_drawdown_pct"] < 2.5


def test_intrabar_stop_loss_priority() -> None:
    runner = _build_runner()
    position = Position(
        symbol=TEST_SYMBOL,
        side=Side.LONG,
        entry_price=100.0,
        quantity=1.0,
        stop_loss=99.0,
        take_profit=101.0,
    )
    runner.position_manager.positions[TEST_SYMBOL] = position
    candle = Candle(
        symbol=TEST_SYMBOL,
        open_time=datetime.utcnow(),
        close_time=datetime.utcnow() + timedelta(minutes=1),
        open=100.0,
        high=101.5,
        low=98.5,
        close=100.2,
        volume=500.0,
    )
    exit_price, reason = runner._protective_hit(position, candle)
    assert exit_price == pytest.approx(position.stop_loss)
    assert reason == "stop_loss"


def _price_frame(rows: int = 150, drift: float = 0.8) -> pd.DataFrame:
    base = datetime.utcnow()
    records = []
    price = 200.0
    for idx in range(rows):
        open_time = base + timedelta(minutes=idx)
        close_time = open_time + timedelta(minutes=1)
        open_price = price
        price += drift + (0.2 if idx % 5 == 0 else 0.0)
        close_price = price
        high = max(open_price, close_price) + 0.3
        low = min(open_price, close_price) - 0.3
        records.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": 2_500.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _run_backtester(cfg, frame: pd.DataFrame, params, symbol: str) -> float:
    filters = SymbolFilters(min_qty=0.001, min_notional=5.0, step_size=0.001, tick_size=0.1, max_leverage=50.0)
    exchange = ExchangeInfoManager(cfg, client=None, prefetched={symbol: filters})
    tester = Backtester(cfg, exchange)
    outcome = tester.run(symbol, TEST_INTERVAL, frame, params)
    return outcome["metrics"]["total_pnl"]


def test_realism_profiles_are_monotonic(tmp_path) -> None:
    cfg = load_config(base_dir=tmp_path)
    ensure_directories(cfg.paths)
    params = build_parameters(cfg)
    frame = _price_frame()
    symbol = TEST_SYMBOL

    def with_level(level: str):
        return replace(cfg, backtest=replace(cfg.backtest, realism_level=level))

    pnl_toy = _run_backtester(with_level("toy"), frame, params, symbol)
    pnl_standard = _run_backtester(with_level("standard"), frame, params, symbol)
    pnl_aggressive = _run_backtester(with_level("aggressive"), frame, params, symbol)

    assert pnl_toy >= pnl_standard - 1e-6
    assert pnl_standard >= pnl_aggressive - 1e-6
