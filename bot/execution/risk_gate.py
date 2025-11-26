from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bot.core.config import BotConfig
from bot.execution.exchange import SymbolFilters
from bot.risk.engine import ExposureState, RiskEngine


@dataclass(slots=True)
class RiskCheck:
    symbol: str
    side: str
    quantity: float
    price: float
    available_balance: float
    equity: float
    exposure: ExposureState
    filters: SymbolFilters | None = None


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    should_flatten: bool
    reason: Optional[str] = None


class RiskGate:
    """Single entry point for live trade approvals."""

    def __init__(self, cfg: BotConfig, engine: RiskEngine) -> None:
        self._cfg = cfg
        self._engine = engine

    @property
    def engine(self) -> RiskEngine:
        return self._engine

    def assess_entry(self, check: RiskCheck) -> RiskDecision:
        allowed, reason = self._engine.can_open_new_trades()
        flatten = self._engine.should_flatten_positions()
        if flatten:
            return RiskDecision(False, True, self._engine.halt_reason or "flatten_required")
        if not allowed:
            return RiskDecision(False, False, reason)
        if check.quantity <= 0 or check.price <= 0:
            return RiskDecision(False, False, "invalid_order")
        adjusted, limit_reason = self._engine.adjust_quantity(
            symbol=check.symbol,
            price=check.price,
            quantity=check.quantity,
            available_balance=check.available_balance,
            equity=check.equity,
            symbol_exposure=check.exposure.per_symbol.get(check.symbol, 0.0),
            total_exposure=check.exposure.total,
            filters=check.filters,
        )
        if adjusted <= 0:
            return RiskDecision(False, False, limit_reason or "risk_limit")
        return RiskDecision(True, False, None)


__all__ = ["RiskGate", "RiskDecision", "RiskCheck"]
