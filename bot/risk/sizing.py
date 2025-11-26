"""Position sizing helpers shared by simulators and live trading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from bot.core.config import BotConfig
from bot.execution.exchange import SymbolFilters
from bot.risk.engine import RiskEngine
from bot.signals.strategies import StrategyParameters


@dataclass(slots=True)
class SizingContext:
    symbol: str
    balance: float
    equity: float
    available_balance: float
    price: float
    params: StrategyParameters
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
        self.min_notional: float = cfg.sizing.min_notional

    def plan_trade(self, ctx: SizingContext, engine: Optional[RiskEngine] = None) -> SizingResult:
        if ctx.price <= 0:
            return SizingResult(0.0, 0.0, True, "invalid_price", False)
        if self._config.risk.max_concurrent_symbols > 0 and not ctx.symbol_already_active:
            if ctx.active_symbols >= self._config.risk.max_concurrent_symbols:
                return SizingResult(0.0, 0.0, True, "max_symbols", False)
        quantity = self._atr_position(ctx)
        if quantity <= 0:
            return SizingResult(0.0, 0.0, True, "no_budget", False)
        notional = quantity * ctx.price
        budget_notional = self._respect_notional_caps(notional, ctx)
        if budget_notional <= 0:
            return SizingResult(0.0, 0.0, True, "min_notional", False)
        quantity = budget_notional / ctx.price
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

    def _atr_position(self, ctx: SizingContext) -> float:
        balance = max(ctx.balance, 0.0)
        if balance <= 0:
            return 0.0
        equity_risk = balance * self._config.risk.per_trade_risk
        if equity_risk <= 0:
            return 0.0
        atr = ctx.volatility.get("atr", 0.0)
        stop_multiple = max(ctx.params.atr_stop, 0.1)
        risk_per_unit = max(atr * stop_multiple, 1e-8)
        return equity_risk / risk_per_unit if risk_per_unit > 0 else 0.0

    def _respect_notional_caps(self, notional: float, ctx: SizingContext) -> float:
        capped = notional
        if ctx.max_notional and ctx.max_notional > 0:
            capped = min(capped, ctx.max_notional)
        max_notional_cfg = self._config.sizing.max_notional
        if max_notional_cfg and max_notional_cfg > 0:
            capped = min(capped, max_notional_cfg)
        if capped < self.min_notional:
            return 0.0
        return capped


__all__ = ["PositionSizer", "SizingContext", "SizingResult"]
