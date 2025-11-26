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
        sizing_builder: Callable[[MarketContext, float, Dict[str, float]], Optional[SizingContext]],
        timestamp_fn: Callable[[pd.Series], str],
        log_sizing_skip: Callable[[MarketContext, Optional[str]], None],
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
        signal = self._latest_signal(frame)
        if signal is None:
            return None
        price = float(latest_row["close"])
        volatility = volatility_snapshot(frame, self._cfg)
        if self._cfg.risk.require_sl_tp and (signal.stop_loss is None or signal.take_profit is None):
            self._log_sizing_skip(self._ctx, "missing_sl_tp")
            return None
        sizing_ctx = self._sizing_builder(self._ctx, price, volatility)
        if sizing_ctx is None:
            return None
        sizing_result = self._sizer.plan_trade(sizing_ctx, self._risk_gate.engine)
        if not sizing_result.accepted:
            self._log_sizing_skip(self._ctx, sizing_result.reason)
            return None
        filters = sizing_ctx.filters or self._exchange_info.get_filters(self._ctx.symbol)
        valid, reason, adjusted_qty = self._exchange_info.validate_order(self._ctx.symbol, price, sizing_result.quantity)
        if not valid or adjusted_qty <= 0:
            self._log_sizing_skip(self._ctx, reason)
            return None
        mta_ok, mta_meta = self._multi_filter.evaluate(self._ctx.symbol, signal.direction)
        if not mta_ok:
            return None
        signal_ok, signal_meta = self._external_gate.evaluate(self._ctx.symbol, signal.direction)
        if not signal_ok:
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
        if not signals:
            return None
        signal = signals[-1]
        if signal.index != len(frame) - 1:
            return None
        return signal


__all__ = ["MarketLoop", "EntryPlan", "ExitPlan"]
