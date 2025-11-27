"""Position sizing helpers shared by simulators and live trading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from bot.core.config import BotConfig
from bot.execution.exchange import SymbolFilters
from bot.risk.engine import RiskEngine
from bot.risk.spec import build_core_risk_config
from bot.signals.strategies import StrategyParameters
from core.risk_shared import apply_notional_limits, stop_based_position_size


@dataclass(slots=True)
class SizingContext:
    symbol: str
    balance: float
    equity: float
    available_balance: float
    price: float
    params: StrategyParameters
    stop_loss: Optional[float]
    volatility: Dict[str, float]
    filters: Optional[SymbolFilters]
    max_notional: Optional[float]
    symbol_exposure: float
    total_exposure: float
    active_symbols: int
    symbol_already_active: bool


@dataclass(slots=True)
class SizingResult:
    quantity: float
    notional: float
    rejected: bool
    reason: Optional[str]
    capped: bool

    @property
    def accepted(self) -> bool:
        return not self.rejected and self.quantity > 0


class PositionSizer:
    def __init__(self, cfg: BotConfig) -> None:
        self._config = cfg
        self._core_risk = build_core_risk_config(cfg)
        self.min_notional = self._core_risk.min_order_notional_usd

    def plan_trade(self, ctx: SizingContext, engine: Optional[RiskEngine] = None) -> SizingResult:
        if ctx.price <= 0:
            return SizingResult(0.0, 0.0, True, "invalid_price", False)
        if self._config.risk.max_concurrent_symbols > 0 and not ctx.symbol_already_active:
            if ctx.active_symbols >= self._config.risk.max_concurrent_symbols:
                return SizingResult(0.0, 0.0, True, "max_symbols", False)
        if ctx.stop_loss is None or ctx.stop_loss == ctx.price:
            return SizingResult(0.0, 0.0, True, "missing_stop", False)
        quantity = stop_based_position_size(
            equity=ctx.equity,
            entry_price=ctx.price,
            stop_price=ctx.stop_loss,
            max_risk_per_trade_pct=self._core_risk.max_risk_per_trade_pct,
            symbol_info=None,
        )
        if quantity <= 0:
            return SizingResult(0.0, 0.0, True, "no_budget", False)
        quantity = apply_notional_limits(
            quantity,
            price=ctx.price,
            min_order_notional=self.min_notional,
            symbol_info=None,
            symbol_cap_usd=self._core_risk.max_symbol_notional_usd or None,
            total_cap_usd=self._core_risk.max_total_notional_usd or None,
            symbol_open_notional=ctx.symbol_exposure,
            total_open_notional=ctx.total_exposure,
        )
        if quantity <= 0:
            return SizingResult(0.0, 0.0, True, "notional_cap", False)
        budget_caps = [ctx.max_notional, self._config.sizing.max_notional]
        for cap in budget_caps:
            if cap and cap > 0:
                cap_qty = cap / ctx.price
                if cap_qty <= 0:
                    return SizingResult(0.0, 0.0, True, "max_notional", False)
                quantity = min(quantity, cap_qty)
        capped = False
        if engine:
            adjusted_qty, reason = engine.adjust_quantity(
                symbol=ctx.symbol,
                price=ctx.price,
                quantity=quantity,
                available_balance=ctx.available_balance,
                equity=ctx.equity,
                symbol_exposure=ctx.symbol_exposure,
                total_exposure=ctx.total_exposure,
                filters=ctx.filters,
            )
            if adjusted_qty <= 0:
                return SizingResult(0.0, 0.0, True, reason or "risk_limit", False)
            capped = adjusted_qty < quantity
            quantity = adjusted_qty
        notional = quantity * ctx.price
        if notional < self.min_notional:
            return SizingResult(0.0, 0.0, True, "min_notional", capped)
        return SizingResult(quantity, notional, False, None, capped)


__all__ = ["PositionSizer", "SizingContext", "SizingResult"]
