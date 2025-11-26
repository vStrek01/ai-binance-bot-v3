"""Margin and exposure risk enforcement for live trading."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

from bot.core.config import BotConfig
from bot.execution.exchange import SymbolFilters
from bot.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ExposureState:
    per_symbol: Dict[str, float]
    total: float


class RiskEngine:
    """Enforces margin, exposure, and exchange sizing constraints."""

    def __init__(self, cfg: BotConfig) -> None:
        self._config = cfg
        self._margin_blocks: Dict[str, Dict[str, float]] = {}
        self._last_free_balance: float = 0.0
        self._pnl_window: Deque[Tuple[float, float]] = deque()
        self._window_realized: float = 0.0
        self._reference_equity: float = 0.0
        self._last_equity: float = 0.0
        self._loss_triggered: bool = False
        self._loss_reason: Optional[str] = None
        self._loss_triggered_at: Optional[float] = None
        self._halt_trading: bool = False
        self._flatten_positions: bool = False
        self._state_version: int = 0

    def on_balance_refresh(self, free_balance: float) -> None:
        """Releases cached warnings once free margin recovers."""
        self._last_free_balance = max(free_balance, 0.0)
        relief = max(self._config.risk.margin_relief_factor, 1.0)
        for symbol, state in list(self._margin_blocks.items()):
            threshold = state.get("balance", 0.0) * relief
            if self._last_free_balance >= threshold > 0:
                self._margin_blocks.pop(symbol, None)

    def should_log_margin_block(self, symbol: str, free_balance: float) -> bool:
        """Throttle repeated margin warnings until balance improves."""
        now = time.time()
        cooldown = max(self._config.risk.margin_warning_cooldown, 1.0)
        state = self._margin_blocks.get(symbol)
        if state:
            last_balance = state.get("balance", 0.0)
            if (
                free_balance < last_balance * self._config.risk.margin_relief_factor
                and (now - state.get("timestamp", 0.0)) < cooldown
            ):
                return False
        self._margin_blocks[symbol] = {"balance": max(free_balance, 0.0), "timestamp": now}
        return True

    def update_equity(self, equity: float) -> None:
        sanitized = max(equity, 0.0)
        self._last_equity = sanitized
        if not self._pnl_window:
            self._reference_equity = sanitized
        self._prune_loss_window(time.time())

    def register_trade(self, pnl: float, equity: float, timestamp: Optional[float] = None) -> None:
        try:
            now = float(timestamp) if timestamp is not None else time.time()
        except (TypeError, ValueError):  # pragma: no cover - defensive
            now = time.time()
        self.update_equity(equity)
        self._pnl_window.append((now, pnl))
        self._window_realized += pnl
        self._prune_loss_window(now)
        if not self._reference_equity:
            self._reference_equity = self._last_equity or max(equity, 0.0)
        self._evaluate_daily_limits(now)

    def can_open_new_trades(self) -> Tuple[bool, Optional[str]]:
        if self._halt_trading:
            return False, self._loss_reason or "risk_halt"
        return True, None

    @property
    def trading_paused(self) -> bool:
        return self._halt_trading

    @property
    def halt_reason(self) -> Optional[str]:
        return self._loss_reason

    def should_flatten_positions(self) -> bool:
        return self._flatten_positions

    def clear_flatten_request(self) -> None:
        self._flatten_positions = False

    def reset_daily_limits(self) -> None:
        self._pnl_window.clear()
        self._reset_loss_window()

    def snapshot(self) -> Dict[str, Any]:
        reference = self._reference_equity or self._last_equity
        loss_amount = max(-self._window_realized, 0.0)
        loss_pct = (loss_amount / reference) if reference > 0 else 0.0
        lookback = max(self._config.risk.daily_loss_lookback_hours, 1)
        return {
            "lookback_hours": lookback,
            "realized": self._window_realized,
            "loss_abs": loss_amount,
            "loss_pct": loss_pct,
            "reference_equity": reference,
            "max_daily_loss_pct": self._config.risk.max_daily_loss_pct,
            "max_daily_loss_abs": self._config.risk.max_daily_loss_abs,
            "trading_paused": self._halt_trading,
            "reason": self._loss_reason,
            "triggered_at": self._loss_triggered_at,
            "flatten_required": self._flatten_positions,
            "stop_trading_on_daily_loss": self._config.risk.stop_trading_on_daily_loss,
            "close_positions_on_daily_loss": self._config.risk.close_positions_on_daily_loss,
            "state_version": self._state_version,
        }

    def _prune_loss_window(self, now: float) -> None:
        lookback_hours = max(self._config.risk.daily_loss_lookback_hours, 1)
        cutoff = now - (lookback_hours * 3600)
        dirty = False
        while self._pnl_window and self._pnl_window[0][0] < cutoff:
            _, pnl = self._pnl_window.popleft()
            self._window_realized -= pnl
            dirty = True
        if dirty and not self._pnl_window:
            self._reset_loss_window()

    def _reset_loss_window(self) -> None:
        self._window_realized = 0.0
        self._reference_equity = self._last_equity
        self._loss_triggered = False
        self._loss_reason = None
        self._loss_triggered_at = None
        self._halt_trading = False
        self._flatten_positions = False
        self._state_version += 1

    def _evaluate_daily_limits(self, now: float) -> None:
        limit_pct = max(self._config.risk.max_daily_loss_pct, 0.0)
        limit_abs = self._config.risk.max_daily_loss_abs or 0.0
        reference = self._reference_equity or self._last_equity
        loss_amount = max(-self._window_realized, 0.0)
        loss_pct = (loss_amount / reference) if reference > 0 else 0.0
        triggered = False
        reason: Optional[str] = None
        if limit_pct > 0 and loss_pct >= limit_pct:
            triggered = True
            reason = "daily_loss_pct"
        if limit_abs > 0 and loss_amount >= limit_abs:
            triggered = True
            reason = reason or "daily_loss_abs"
        if not triggered:
            if self._loss_triggered:
                self._state_version += 1
            self._loss_triggered = False
            self._loss_reason = None
            self._loss_triggered_at = None
            self._halt_trading = False
            self._flatten_positions = False
            return
        if not self._loss_triggered:
            self._loss_triggered = True
            self._loss_reason = reason
            self._loss_triggered_at = now
            self._state_version += 1
            logger.error(
                "Daily loss limit breached: loss=%.2f pct=%.4f (stop=%s flatten=%s)",
                loss_amount,
                loss_pct,
                self._config.risk.stop_trading_on_daily_loss,
                self._config.risk.close_positions_on_daily_loss,
            )
        self._halt_trading = self._config.risk.stop_trading_on_daily_loss or self._halt_trading
        if self._config.risk.close_positions_on_daily_loss:
            self._flatten_positions = True

    def adjust_quantity(
        self,
        *,
        symbol: str,
        price: float,
        quantity: float,
        available_balance: float,
        equity: float,
        symbol_exposure: float,
        total_exposure: float,
        filters: Optional[SymbolFilters],
    ) -> Tuple[float, Optional[str]]:
        """Scale or block quantity so that all hard limits remain satisfied."""
        if price <= 0 or quantity <= 0:
            return 0.0, "invalid_request"
        notional = quantity * price
        if notional <= 0:
            return 0.0, "invalid_request"

        margin_cap = self._max_margin_notional(available_balance, filters)
        scale = 1.0
        limiting_reason: Optional[str] = None

        if margin_cap <= 0:
            return 0.0, "margin"
        if notional > margin_cap:
            scale = min(scale, margin_cap / notional)
            limiting_reason = "margin"

        symbol_cap = self._symbol_cap(equity)
        if symbol_cap > 0:
            remaining = max(symbol_cap - symbol_exposure, 0.0)
            if remaining <= 0:
                return 0.0, "symbol_cap"
            if notional > remaining:
                ratio = remaining / notional
                if ratio < scale:
                    limiting_reason = "symbol_cap"
                scale = min(scale, ratio)

        portfolio_cap = self._portfolio_cap(equity)
        if portfolio_cap > 0:
            remaining_total = max(portfolio_cap - total_exposure, 0.0)
            if remaining_total <= 0:
                return 0.0, "portfolio_cap"
            if notional > remaining_total:
                ratio = remaining_total / notional
                if ratio < scale:
                    limiting_reason = "portfolio_cap"
                scale = min(scale, ratio)

        scaled_qty = quantity * max(min(scale, 1.0), 0.0)
        if scaled_qty <= 0:
            return 0.0, limiting_reason or "scaled_zero"

        if filters:
            scaled_qty = filters.adjust_quantity(scaled_qty)
            price = filters.adjust_price(price)
            if scaled_qty < filters.min_qty:
                return 0.0, "min_qty"
            if scaled_qty * price < filters.min_notional:
                return 0.0, "min_notional"

        return scaled_qty, limiting_reason

    def compute_exposure(self, exchange_positions: Dict[str, Dict[str, float]]) -> ExposureState:
        per_symbol: Dict[str, float] = {}
        total = 0.0
        for symbol, payload in exchange_positions.items():
            qty = abs(float(payload.get("quantity", 0.0)))
            mark_price = float(payload.get("mark_price") or payload.get("entry_price") or 0.0)
            notional = qty * max(mark_price, 0.0)
            if notional <= 0:
                continue
            per_symbol[symbol] = notional
            total += notional
        return ExposureState(per_symbol=per_symbol, total=total)

    def _max_margin_notional(self, available_balance: float, filters: Optional[SymbolFilters]) -> float:
        usable = max(available_balance - self._config.risk.min_free_margin, 0.0)
        usable *= max(min(1.0 - self._config.risk.margin_buffer, 1.0), 0.0)
        leverage_cap = self._config.risk.leverage
        if filters:
            leverage_cap = min(leverage_cap, filters.max_leverage or leverage_cap)
        return usable * leverage_cap * max(min(self._config.runtime.max_margin_utilization, 0.99), 0.05)

    def _symbol_cap(self, equity: float) -> float:
        if self._config.risk.max_symbol_exposure <= 0:
            return 0.0
        return max(equity, 0.0) * self._config.risk.max_symbol_exposure

    def _portfolio_cap(self, equity: float) -> float:
        if self._config.risk.max_account_exposure <= 0:
            return 0.0
        return max(equity, 0.0) * self._config.risk.max_account_exposure


__all__ = ["ExposureState", "RiskEngine"]
