from __future__ import annotations

from dataclasses import dataclass

from bot.core.config import BotConfig
from bot.execution.exchange import SymbolFilters
from bot.risk.engine import ExposureState, OpenTradeRequest, RiskDecision, RiskEngine


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


class RiskGate:
    """Single entry point for live trade approvals."""

    def __init__(self, cfg: BotConfig, engine: RiskEngine) -> None:
        self._cfg = cfg
        self._engine = engine

    @property
    def engine(self) -> RiskEngine:
        return self._engine

    def assess_entry(self, check: RiskCheck) -> RiskDecision:
        request = OpenTradeRequest(
            symbol=check.symbol,
            side=check.side,
            quantity=check.quantity,
            price=check.price,
            available_balance=check.available_balance,
            equity=check.equity,
            exposure=check.exposure,
            filters=check.filters,
        )
        return self._engine.evaluate_open(request)


__all__ = ["RiskGate", "RiskDecision", "RiskCheck"]
