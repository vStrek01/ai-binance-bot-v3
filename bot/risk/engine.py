"""Margin and exposure risk enforcement for live trading."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from bot.core.config import BotConfig, RiskConfig
from bot.execution.exchange import SymbolFilters
from bot.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ExposureState:
    per_symbol: Dict[str, float]
    total: float


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    should_flatten: bool
    reason: Optional[str] = None


@dataclass(slots=True)
class RiskState:
    equity: float
    pnl_today: float
    loss_abs: float
    loss_pct: float
    max_daily_loss_pct: float
    max_daily_loss_abs: float | None
    loss_streak: int
    max_consecutive_losses: int
    trading_mode: "TradingMode"
    flatten_required: bool
    last_trigger_reason: Optional[str]
    last_triggered_at: Optional[float]
    reference_equity: float


@dataclass(slots=True)
class RiskEvent:
    timestamp: float
    event_type: str
    reason: Optional[str]
    equity: float
    pnl_today: float
    loss_abs: float
    loss_pct: float
    symbol: Optional[str] = None


@dataclass(slots=True)
class TradeEvent:
    pnl: float
    equity: float
    timestamp: Optional[float] = None
    symbol: Optional[str] = None


@dataclass(slots=True)
class OpenTradeRequest:
    symbol: str
    side: str
    quantity: float
    price: float
    available_balance: float
    equity: float
    exposure: ExposureState
    filters: SymbolFilters | None = None


@dataclass(slots=True)
class CloseTradeRequest:
    symbol: str
    quantity: float
    price: float
    equity: float
    exposure: ExposureState
    filters: SymbolFilters | None = None


class TradingMode(Enum):
    NORMAL = "normal"
    HALTED_DAILY_LOSS = "halted_daily_loss"
    HALTED_MANUAL = "halted_manual"
    HALTED_LOSS_STREAK = "halted_loss_streak"


class RiskEngine:
    """Enforces margin, exposure, and exchange sizing constraints."""

    _AUDIT_LIMIT = 200

    def __init__(self, cfg: BotConfig) -> None:
        self._config = self._clamp_config(cfg)
        self._margin_blocks: Dict[str, Dict[str, float]] = {}
        self._last_free_balance: float = 0.0
        self._pnl_window: Deque[Tuple[float, float]] = deque()
        self._window_realized: float = 0.0
        self._reference_equity: float = 0.0
        self._last_equity: float = 0.0
        self._trading_mode: TradingMode = TradingMode.NORMAL
        self._halt_reason: Optional[str] = None
        self._loss_triggered_at: Optional[float] = None
        self._flatten_positions: bool = False
        self._audit: Deque[RiskEvent] = deque(maxlen=self._AUDIT_LIMIT)
        self._state_version: int = 0
        self._loss_streak: int = 0
        self._max_loss_streak: int = max(int(cfg.risk.max_consecutive_losses or 0), 0)
        self._demo_mode = cfg.run_mode == "demo-live"
        self._abs_symbol_cap = max(float(cfg.risk.max_notional_per_symbol or 0.0), 0.0)
        self._abs_global_cap = max(float(cfg.risk.max_notional_global or 0.0), 0.0)

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

    def register_trade(self, trade: TradeEvent) -> None:
        now = self._sanitize_timestamp(trade.timestamp)
        self.update_equity(trade.equity)
        self._pnl_window.append((now, trade.pnl))
        self._window_realized += trade.pnl
        if trade.pnl < 0:
            self._loss_streak += 1
        elif trade.pnl > 0:
            self._loss_streak = 0
        self._prune_loss_window(now)
        if not self._reference_equity:
            self._reference_equity = self._last_equity or max(trade.equity, 0.0)
        if self._max_loss_streak > 0 and self._loss_streak >= self._max_loss_streak:
            self._trigger_loss_streak(now, symbol=trade.symbol)
        self._evaluate_daily_limits(now, symbol=trade.symbol)

    def evaluate_open(self, request: OpenTradeRequest) -> RiskDecision:
        self.update_equity(request.equity)
        if self._trading_mode != TradingMode.NORMAL:
            decision = RiskDecision(False, self._flatten_positions, self._halt_reason)
            self._record_event("open_blocked", request.symbol, decision.reason)
            return decision
        if request.quantity <= 0 or request.price <= 0:
            decision = RiskDecision(False, False, "invalid_request")
            self._record_event("open_blocked", request.symbol, decision.reason)
            return decision
        adjusted, limit_reason = self.adjust_quantity(
            symbol=request.symbol,
            price=request.price,
            quantity=request.quantity,
            available_balance=request.available_balance,
            equity=request.equity,
            symbol_exposure=request.exposure.per_symbol.get(request.symbol, 0.0),
            total_exposure=request.exposure.total,
            filters=request.filters,
        )
        if adjusted <= 0:
            decision = RiskDecision(False, False, limit_reason or "risk_limit")
            self._record_event("open_blocked", request.symbol, decision.reason)
            return decision
        return RiskDecision(True, False, None)

    def evaluate_close(self, request: CloseTradeRequest) -> RiskDecision:
        self.update_equity(request.equity)
        if request.quantity <= 0 or request.price <= 0:
            decision = RiskDecision(False, False, "invalid_request")
            self._record_event("close_blocked", request.symbol, decision.reason)
            return decision
        if self._flatten_positions or self._trading_mode != TradingMode.NORMAL:
            return RiskDecision(True, True, self._halt_reason)
        return RiskDecision(True, False, None)

    def current_state(self) -> RiskState:
        reference = self._reference_equity or self._last_equity
        loss_amount = max(-self._window_realized, 0.0)
        loss_pct = (loss_amount / reference) if reference > 0 else 0.0
        return RiskState(
            equity=self._last_equity,
            pnl_today=self._window_realized,
            loss_abs=loss_amount,
            loss_pct=loss_pct,
            max_daily_loss_pct=self._config.risk.max_daily_loss_pct,
            max_daily_loss_abs=self._config.risk.max_daily_loss_abs,
            loss_streak=self._loss_streak,
            max_consecutive_losses=self._max_loss_streak,
            trading_mode=self._trading_mode,
            flatten_required=self._flatten_positions,
            last_trigger_reason=self._halt_reason,
            last_triggered_at=self._loss_triggered_at,
            reference_equity=reference,
        )

    def reset_daily_limits(self) -> None:
        self._pnl_window.clear()
        self._reset_loss_window()

    def snapshot(self) -> Dict[str, Any]:
        state = self.current_state()
        return {
            "equity": state.equity,
            "pnl_today": state.pnl_today,
            "loss_abs": state.loss_abs,
            "loss_pct": state.loss_pct,
            "loss_streak": state.loss_streak,
            "max_consecutive_losses": state.max_consecutive_losses,
            "max_daily_loss_pct": state.max_daily_loss_pct,
            "max_daily_loss_abs": state.max_daily_loss_abs,
            "trading_mode": state.trading_mode.value,
            "trading_paused": state.trading_mode != TradingMode.NORMAL,
            "flatten_required": state.flatten_required,
            "reason": state.last_trigger_reason,
            "triggered_at": state.last_triggered_at,
            "reference_equity": state.reference_equity,
            "state_version": self._state_version,
            "recent_events": [
                {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "reason": event.reason,
                    "symbol": event.symbol,
                    "equity": event.equity,
                    "pnl_today": event.pnl_today,
                    "loss_abs": event.loss_abs,
                    "loss_pct": event.loss_pct,
                }
                for event in self.get_recent_events(limit=20)
            ],
        }

    def can_open_new_trades(self) -> Tuple[bool, Optional[str]]:
        if self._trading_mode != TradingMode.NORMAL:
            return False, self._halt_reason or self._trading_mode.value
        return True, None

    @property
    def trading_paused(self) -> bool:
        return self._trading_mode != TradingMode.NORMAL

    @property
    def halt_reason(self) -> Optional[str]:
        return self._halt_reason

    def should_flatten_positions(self) -> bool:
        return self._flatten_positions

    def clear_flatten_request(self) -> None:
        self._flatten_positions = False

    def set_manual_halt(self, reason: str) -> None:
        if self._trading_mode == TradingMode.HALTED_MANUAL:
            return
        self._trading_mode = TradingMode.HALTED_MANUAL
        self._halt_reason = reason
        self._loss_triggered_at = time.time()
        self._record_event("manual_halt", None, reason)
        self._state_version += 1

    def clear_manual_halt(self) -> None:
        if self._trading_mode != TradingMode.HALTED_MANUAL:
            return
        self._reset_loss_window()

    def get_recent_events(self, limit: int = 100) -> List[RiskEvent]:
        if limit <= 0:
            return []
        return list(self._audit)[-limit:]

    def _prune_loss_window(self, now: float) -> None:
        lookback_hours = max(self._config.risk.daily_loss_lookback_hours, 1)
        cutoff = now - (lookback_hours * 3600)
        while self._pnl_window and self._pnl_window[0][0] < cutoff:
            _, pnl = self._pnl_window.popleft()
            self._window_realized -= pnl

    def _reset_loss_window(self) -> None:
        self._window_realized = 0.0
        self._reference_equity = self._last_equity
        self._trading_mode = TradingMode.NORMAL
        self._halt_reason = None
        self._loss_triggered_at = None
        self._flatten_positions = False
        self._loss_streak = 0
        self._state_version += 1

    def _evaluate_daily_limits(self, now: float, *, symbol: Optional[str]) -> None:
        if self._trading_mode == TradingMode.HALTED_DAILY_LOSS:
            return
        limit_pct = max(self._config.risk.max_daily_loss_pct, 0.0)
        limit_abs = self._config.risk.max_daily_loss_abs or 0.0
        reference = self._reference_equity or self._last_equity
        loss_amount = max(-self._window_realized, 0.0)
        loss_pct = (loss_amount / reference) if reference > 0 else 0.0
        trigger: Optional[str] = None
        if limit_pct > 0 and loss_pct >= limit_pct:
            trigger = "daily_loss_pct"
        if limit_abs > 0 and loss_amount >= limit_abs:
            trigger = trigger or "daily_loss_abs"
        if trigger is None:
            return
        self._trading_mode = TradingMode.HALTED_DAILY_LOSS
        self._halt_reason = trigger
        self._loss_triggered_at = now
        self._flatten_positions = self._config.risk.close_positions_on_daily_loss or self._flatten_positions
        self._state_version += 1
        self._record_event("daily_loss_triggered", symbol, trigger, loss_amount, loss_pct)
        logger.error(
            "Daily loss limit breached: loss=%.2f pct=%.4f (mode=%s flatten=%s)",
            loss_amount,
            loss_pct,
            self._trading_mode.value,
            self._flatten_positions,
        )

    def _trigger_loss_streak(self, now: float, *, symbol: Optional[str]) -> None:
        if self._max_loss_streak <= 0:
            return
        if self._trading_mode not in {TradingMode.NORMAL, TradingMode.HALTED_LOSS_STREAK}:
            return
        if self._trading_mode == TradingMode.HALTED_LOSS_STREAK and self._halt_reason == "loss_streak":
            return
        self._trading_mode = TradingMode.HALTED_LOSS_STREAK
        self._halt_reason = "loss_streak"
        self._loss_triggered_at = now
        self._state_version += 1
        self._record_event("loss_streak_triggered", symbol, self._halt_reason)

    def _record_event(
        self,
        event_type: str,
        symbol: Optional[str],
        reason: Optional[str],
        loss_abs: Optional[float] = None,
        loss_pct: Optional[float] = None,
    ) -> None:
        state = self.current_state()
        loss_abs = loss_abs if loss_abs is not None else state.loss_abs
        loss_pct = loss_pct if loss_pct is not None else state.loss_pct
        event = RiskEvent(
            timestamp=time.time(),
            event_type=event_type,
            reason=reason,
            equity=state.equity,
            pnl_today=state.pnl_today,
            loss_abs=loss_abs,
            loss_pct=loss_pct,
            symbol=symbol,
        )
        self._audit.append(event)
        logger.warning(
            "risk_event", extra={"event_type": event_type, "reason": reason, "symbol": symbol, "equity": state.equity}
        )

    def _sanitize_timestamp(self, timestamp: Optional[float]) -> float:
        try:
            return float(timestamp) if timestamp is not None else time.time()
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return time.time()

    def _clamp_config(self, cfg: BotConfig) -> BotConfig:
        risk = cfg.risk
        clamps: Dict[str, float] = {}
        if cfg.runtime.live_trading or not cfg.runtime.dry_run:
            leverage_cap = 25.0
            max_daily_loss_pct_cap = 0.2
            if risk.leverage > leverage_cap:
                clamps["leverage"] = leverage_cap
            if risk.max_daily_loss_pct > max_daily_loss_pct_cap:
                clamps["max_daily_loss_pct"] = max_daily_loss_pct_cap
        if clamps:
            logger.warning("Risk config clamped for safety: %s", clamps)
            risk = risk.model_copy(update=clamps)
        return cfg.model_copy(update={"risk": risk})

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

        if self._abs_symbol_cap > 0:
            remaining_abs = max(self._abs_symbol_cap - symbol_exposure, 0.0)
            if remaining_abs <= 0:
                return 0.0, "symbol_abs_cap"
            if notional > remaining_abs:
                if self._demo_mode:
                    return 0.0, "symbol_abs_cap"
                ratio = remaining_abs / notional
                if ratio < scale:
                    limiting_reason = "symbol_abs_cap"
                scale = min(scale, ratio)

        if self._abs_global_cap > 0:
            remaining_abs_total = max(self._abs_global_cap - total_exposure, 0.0)
            if remaining_abs_total <= 0:
                return 0.0, "portfolio_abs_cap"
            if notional > remaining_abs_total:
                if self._demo_mode:
                    return 0.0, "portfolio_abs_cap"
                ratio = remaining_abs_total / notional
                if ratio < scale:
                    limiting_reason = "portfolio_abs_cap"
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


__all__ = [
    "CloseTradeRequest",
    "ExposureState",
    "OpenTradeRequest",
    "RiskDecision",
    "RiskEngine",
    "RiskEvent",
    "RiskState",
    "TradeEvent",
    "TradingMode",
]
