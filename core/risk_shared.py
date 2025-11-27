from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Protocol

from exchange.symbols import SymbolInfo


class RiskLimits(Protocol):
    max_daily_drawdown_pct: float
    max_consecutive_losses: int
    max_trades_per_day: int
    max_commission_pct_per_day: float
    min_order_notional_usd: float
    max_symbol_notional_usd: float
    max_total_notional_usd: float
    max_risk_per_trade_pct: float


@dataclass(slots=True)
class DailyRiskState:
    session_date: date
    starting_equity: float
    realized_pnl: float = 0.0
    trades_count: int = 0
    commission_paid_pct: float = 0.0
    consecutive_losses: int = 0

    def drawdown_pct(self, *, current_equity: Optional[float] = None) -> float:
        if self.starting_equity <= 0:
            return 0.0
        reference = current_equity if current_equity is not None else (self.starting_equity + self.realized_pnl)
        dd = (self.starting_equity - max(reference, 0.0)) / self.starting_equity * 100.0
        return max(dd, 0.0)


def stop_based_position_size(
    *,
    equity: float,
    entry_price: float,
    stop_price: float,
    max_risk_per_trade_pct: float,
    symbol_info: Optional[SymbolInfo] = None,
) -> float:
    risk_budget = max(equity, 0.0) * max(max_risk_per_trade_pct, 0.0)
    distance = abs(entry_price - stop_price)
    if distance <= 0:
        return 0.0
    qty = risk_budget / distance
    if symbol_info:
        qty = symbol_info.round_qty(qty)
    return max(qty, 0.0)


def apply_notional_limits(
    qty: float,
    *,
    price: float,
    min_order_notional: float,
    symbol_info: Optional[SymbolInfo] = None,
    symbol_cap_usd: Optional[float] = None,
    total_cap_usd: Optional[float] = None,
    symbol_open_notional: float = 0.0,
    total_open_notional: float = 0.0,
) -> float:
    if qty <= 0 or price <= 0:
        return 0.0
    notional = qty * price
    if notional < min_order_notional:
        return 0.0
    if symbol_cap_usd and symbol_cap_usd > 0:
        remaining_symbol = max(symbol_cap_usd - symbol_open_notional, 0.0)
        if remaining_symbol <= 0:
            return 0.0
        qty = min(qty, remaining_symbol / price)
    if total_cap_usd and total_cap_usd > 0:
        remaining_total = max(total_cap_usd - total_open_notional, 0.0)
        if remaining_total <= 0:
            return 0.0
        qty = min(qty, remaining_total / price)
    if symbol_info:
        qty = symbol_info.round_qty(qty)
    if qty * price < min_order_notional:
        return 0.0
    if symbol_info and not symbol_info.validate_notional(price, qty):
        return 0.0
    return max(qty, 0.0)


def evaluate_kill_switch(
    config: RiskLimits,
    state: Optional[DailyRiskState],
    *,
    equity: Optional[float],
    relaxed_drawdown: bool = False,
) -> list[str]:
    if state is None:
        return []
    reasons: list[str] = []
    if not relaxed_drawdown:
        limit = max(config.max_daily_drawdown_pct, 0.0) * 100.0
        if limit > 0:
            dd = state.drawdown_pct(current_equity=equity)
            if dd >= limit:
                reasons.append("daily_drawdown")
    if config.max_consecutive_losses > 0 and state.consecutive_losses >= config.max_consecutive_losses:
        reasons.append("consecutive_losses")
    if config.max_trades_per_day > 0 and state.trades_count >= config.max_trades_per_day:
        reasons.append("trade_limit")
    if (
        config.max_commission_pct_per_day > 0
        and state.commission_paid_pct >= config.max_commission_pct_per_day
    ):
        reasons.append("commission")
    return reasons


__all__ = [
    "DailyRiskState",
    "apply_notional_limits",
    "evaluate_kill_switch",
    "stop_based_position_size",
]
