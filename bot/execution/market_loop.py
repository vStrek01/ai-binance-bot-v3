from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import pandas as pd

from bot.core.config import BotConfig
from bot.execution.order_manager import OrderRequest
from bot.execution.risk_gate import RiskCheck, RiskGate
from bot.exchange_info import ExchangeInfoManager
from bot.execution.runners import MarketContext, PaperPosition
from bot.risk import ExternalSignalGate, MultiTimeframeFilter, PositionSizer, volatility_snapshot
from bot.risk.engine import ExposureState
from bot.risk.sizing import SizingContext
from bot.signals.strategies import StrategySignal
from infra.alerts import send_alert
from infra.logging import log_event
from exchange.data_health import DataHealthMonitor, DataHealthStatus, get_data_health_monitor


@dataclass(slots=True)
class EntryPlan:
    position: PaperPosition
    signal: StrategySignal
    order_request: OrderRequest


@dataclass(slots=True)
class ExitPlan:
    side: str
    quantity: float
    price: float
    reason: str


class MarketLoop:
    """Encapsulates per-symbol live decision making."""

    def __init__(
        self,
        ctx: MarketContext,
        cfg: BotConfig,
        *,
        risk_gate: RiskGate,
        sizer: PositionSizer,
        multi_filter: MultiTimeframeFilter,
        external_gate: ExternalSignalGate,
        exchange_info: ExchangeInfoManager,
        sizing_builder: Callable[[MarketContext, float, Dict[str, float], Optional[float]], Optional[SizingContext]],
        timestamp_fn: Callable[[pd.Series], str],
        log_sizing_skip: Callable[[MarketContext, Optional[str]], None],
        data_health: Optional[DataHealthMonitor] = None,
    ) -> None:
        self._ctx = ctx
        self._cfg = cfg
        self._risk_gate = risk_gate
        self._sizer = sizer
        self._multi_filter = multi_filter
        self._external_gate = external_gate
        self._exchange_info = exchange_info
        self._sizing_builder = sizing_builder
        self._timestamp_fn = timestamp_fn
        self._log_sizing_skip = log_sizing_skip
        self._data_health = data_health or get_data_health_monitor()

    def plan_entry(
        self,
        frame: pd.DataFrame,
        latest_row: pd.Series,
        *,
        balance: float,
        equity: float,
        available_balance: float,
        exposure: ExposureState,
    ) -> Optional[EntryPlan]:
        if not self._data_health_allows_trade(stage="entry"):
            return None
        signal = self._latest_signal(frame)
        if signal is None:
            return None
        price = float(latest_row["close"])
        volatility = volatility_snapshot(frame, self._cfg)
        if self._cfg.risk.require_sl_tp and (signal.stop_loss is None or signal.take_profit is None):
            self._log_veto("risk_precheck", "missing_sl_tp")
            self._log_sizing_skip(self._ctx, "missing_sl_tp")
            return None
        sizing_ctx = self._sizing_builder(self._ctx, price, volatility, signal.stop_loss)
        if sizing_ctx is None:
            self._log_veto("sizing", "context_unavailable")
            return None
        sizing_result = self._sizer.plan_trade(sizing_ctx, self._risk_gate.engine)
        if not sizing_result.accepted:
            self._log_sizing_skip(self._ctx, sizing_result.reason)
            self._log_veto("sizing", sizing_result.reason or "rejected")
            return None
        filters = sizing_ctx.filters or self._exchange_info.get_filters(self._ctx.symbol)
        valid, reason, adjusted_qty = self._exchange_info.validate_order(self._ctx.symbol, price, sizing_result.quantity)
        if not valid or adjusted_qty <= 0:
            self._log_sizing_skip(self._ctx, reason)
            self._log_veto("exchange_filters", reason or "invalid_order", requested_qty=sizing_result.quantity)
            return None
        mta_ok, mta_meta = self._multi_filter.evaluate(self._ctx.symbol, signal.direction)
        if not mta_ok:
            self._log_veto("multi_timeframe", "alignment_blocked")
            return None
        signal_ok, signal_meta = self._external_gate.evaluate(self._ctx.symbol, signal.direction)
        if not signal_ok:
            self._log_veto("external_signals", "sentiment_blocked")
            return None
        side = "BUY" if signal.direction == 1 else "SELL"
        risk_decision = self._risk_gate.assess_entry(
            RiskCheck(
                symbol=self._ctx.symbol,
                side=side,
                quantity=adjusted_qty,
                price=price,
                available_balance=available_balance,
                equity=equity,
                exposure=exposure,
                filters=filters,
            )
        )
        if not risk_decision.allowed:
            self._log_sizing_skip(self._ctx, risk_decision.reason)
            self._log_veto("risk_gate", risk_decision.reason or "blocked", available_balance=available_balance)
            return None
        opened_at = self._timestamp_fn(latest_row)
        position = PaperPosition(
            symbol=self._ctx.symbol,
            timeframe=self._ctx.timeframe,
            direction=signal.direction,
            quantity=adjusted_qty,
            entry_price=price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            opened_at=opened_at,
            metadata={"indicators": getattr(signal, "indicators", {})},
        )
        position.metadata.setdefault("sizing", {})
        position.metadata["sizing"].update(
            {
                "mode": self._cfg.sizing.mode,
                "risk_per_trade": balance * self._cfg.risk.per_trade_risk,
                "volatility": volatility,
            }
        )
        position.metadata["mta"] = mta_meta
        position.metadata["external_signals"] = signal_meta
        order_request = OrderRequest(
            symbol=self._ctx.symbol,
            side=side,
            quantity=adjusted_qty,
            order_type="MARKET",
            reduce_only=False,
            price=price,
            filters=filters,
            tag="entry",
        )
        self._log_decision(position, signal, volatility)
        return EntryPlan(position=position, signal=signal, order_request=order_request)

    def plan_exit(self, position: PaperPosition, row: pd.Series) -> Optional[ExitPlan]:
        high = float(row["high"])
        low = float(row["low"])
        exit_price: Optional[float] = None
        exit_reason = "open"
        if position.direction == 1:
            if low <= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = "stop"
            elif high >= position.take_profit:
                exit_price = position.take_profit
                exit_reason = "target"
        else:
            if high >= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = "stop"
            elif low <= position.take_profit:
                exit_price = position.take_profit
                exit_reason = "target"
        if exit_price is None:
            return None
        side = "SELL" if position.direction == 1 else "BUY"
        return ExitPlan(side=side, quantity=position.quantity, price=exit_price, reason=exit_reason)

    def _latest_signal(self, frame: pd.DataFrame) -> Optional[StrategySignal]:
        signals = self._ctx.strategy.generate_signals(frame)
        snapshot = None
        if hasattr(self._ctx.strategy, "latest_snapshot"):
            snapshot = self._ctx.strategy.latest_snapshot()
        signal: Optional[StrategySignal] = None
        if signals:
            candidate = signals[-1]
            if candidate.index == len(frame) - 1:
                signal = candidate
        self._log_tick(snapshot, signal)
        return signal

    def _telemetry_enabled(self) -> bool:
        return self._ctx.run_mode in {"demo-live", "live"}

    def _log_tick(self, snapshot: Optional[Dict[str, float]], signal: Optional[StrategySignal]) -> None:
        if not self._telemetry_enabled():
            return
        payload: Dict[str, Optional[float | str]] = {
            "run_mode": self._ctx.run_mode,
            "symbol": self._ctx.symbol,
            "interval": self._ctx.timeframe,
            "state": "FLAT" if signal is None else ("LONG" if signal.direction == 1 else "SHORT"),
        }
        if snapshot:
            payload.update(
                {
                    "last_close": snapshot.get("last_close"),
                    "ema_fast": snapshot.get("ema_fast"),
                    "ema_slow": snapshot.get("ema_slow"),
                    "rsi": snapshot.get("rsi"),
                }
            )
        log_event("STRATEGY_TICK", **payload)

    def _log_decision(self, position: PaperPosition, signal: StrategySignal, volatility: Dict[str, float]) -> None:
        if not self._telemetry_enabled():
            return
        decision = signal.to_decision(symbol=self._ctx.symbol, interval=self._ctx.timeframe)
        payload = {
            "run_mode": self._ctx.run_mode,
            **decision,
            "size": position.quantity,
            "volatility": volatility,
        }
        log_event("STRATEGY_DECISION", **payload)

    def _log_veto(self, stage: str, reason: Optional[str], **details: object) -> None:
        if not self._telemetry_enabled():
            return
        payload: Dict[str, object] = {
            "run_mode": self._ctx.run_mode,
            "symbol": self._ctx.symbol,
            "interval": self._ctx.timeframe,
            "stage": stage,
            "reason": reason or stage,
        }
        payload.update(details)
        log_event("STRATEGY_VETO", **payload)

    def _data_health_allows_trade(self, *, stage: str) -> bool:
        if self._ctx.run_mode == "backtest" or not self._data_health:
            return True
        status = self._data_health.is_data_stale(self._ctx.symbol, self._ctx.timeframe)
        if status.healthy or status.last_update is None:
            return True
        self._data_health.is_healthy(self._ctx.symbol, self._ctx.timeframe)
        self._log_data_unhealthy(stage, status)
        return False

    def _log_data_unhealthy(self, stage: str, status: DataHealthStatus) -> None:
        payload = {
            "run_mode": self._ctx.run_mode,
            "symbol": self._ctx.symbol,
            "interval": self._ctx.timeframe,
            "stage": stage,
            "source": "market_loop",
            "seconds_since_update": status.seconds_since_update,
            "threshold_seconds": status.threshold_seconds,
        }
        log_event("DATA_UNHEALTHY", **payload)
        send_alert(
            "DATA_UNHEALTHY",
            severity="warning",
            message="Market loop blocked due to stale data",
            **payload,
        )


__all__ = ["MarketLoop", "EntryPlan", "ExitPlan"]
