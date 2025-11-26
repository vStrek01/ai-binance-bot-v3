from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SafetyLimits:
    """Static guard-rail values pulled from configuration."""

    max_daily_drawdown_pct: float
    max_total_notional_usd: float
    max_consecutive_losses: int


@dataclass(slots=True)
class SafetyState:
    """Snapshot of the bot's current risk posture."""

    daily_start_equity: float
    current_equity: float
    consecutive_losses: int


def check_kill_switch(limits: SafetyLimits, state: SafetyState, *, return_reason: bool = False):
    """Return True (or True, reason) when trading must halt under the configured safety limits."""

    if state.daily_start_equity <= 0:
        drawdown_pct = 0.0
    else:
        drawdown_pct = (state.current_equity - state.daily_start_equity) / state.daily_start_equity * 100

    if limits.max_daily_drawdown_pct > 0 and drawdown_pct <= -limits.max_daily_drawdown_pct:
        return (True, "DAILY_DRAWDOWN") if return_reason else True

    if limits.max_consecutive_losses > 0 and state.consecutive_losses >= limits.max_consecutive_losses:
        return (True, "CONSECUTIVE_LOSSES") if return_reason else True

    return (False, None) if return_reason else False


def cap_notional(
    requested_qty: float,
    price: float,
    *,
    max_symbol_notional: float,
    max_total_notional: float,
    current_symbol_notional: float,
    current_total_notional: float,
) -> float:
    """Reduce order quantity so per-symbol and global notionals stay within limits."""

    if requested_qty <= 0 or price <= 0:
        return 0.0

    requested_notional = abs(requested_qty * price)

    if max_symbol_notional <= 0:
        allowed_symbol = requested_notional
    else:
        allowed_symbol = max(0.0, max_symbol_notional - current_symbol_notional)

    if max_total_notional <= 0:
        allowed_global = requested_notional
    else:
        allowed_global = max(0.0, max_total_notional - current_total_notional)

    allowed_notional = min(requested_notional, allowed_symbol, allowed_global)
    if allowed_notional <= 0:
        return 0.0

    capped_qty = allowed_notional / price
    return capped_qty
