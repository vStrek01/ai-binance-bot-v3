from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from core.models import MarketState, OrderRequest, RiskConfig
from exchange.symbols import SymbolInfo, SymbolResolver
from infra.logging import logger, log_event


@dataclass(slots=True)
class DailyRiskState:
    session_date: date
    starting_equity: float
    realized_pnl: float = 0.0
    trades_count: int = 0
    commission_paid_pct: float = 0.0
    consecutive_losses: int = 0

    @property
    def drawdown_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        dd = -self.realized_pnl / self.starting_equity * 100.0
        return max(dd, 0.0)


class RiskManager:
    def __init__(
        self,
        config: RiskConfig,
        *,
        symbol_resolver: SymbolResolver | None = None,
        relaxed_drawdown: bool = False,
        safety_limits: Any | None = None,
    ) -> None:
        self.config = config
        self.symbol_resolver = symbol_resolver
        self.relaxed_drawdown = relaxed_drawdown
        self._last_reset: Optional[datetime] = None
        self._daily_state: Optional[DailyRiskState] = None
        self.kill_switch_active = False
        self._legacy_safety_limits = safety_limits  # kept for compatibility; dedicated risk controls supersede it

    @property
    def daily_start_equity(self) -> Optional[float]:
        return self._daily_state.starting_equity if self._daily_state else None

    @daily_start_equity.setter
    def daily_start_equity(self, value: float) -> None:  # pragma: no cover - compatibility shim
        if self._daily_state:
            self._daily_state.starting_equity = value
        else:
            self._daily_state = DailyRiskState(session_date=date.today(), starting_equity=value)

    def enable_backtest_mode(self) -> None:
        self.relaxed_drawdown = True

    def reset_day_if_needed(self, equity: float) -> None:
        today = date.today()
        if self._daily_state is None or self._daily_state.session_date != today:
            self._daily_state = DailyRiskState(session_date=today, starting_equity=equity)
            self.kill_switch_active = False
            self._last_reset = datetime.utcnow()
            logger.info("Daily risk state initialized", extra={"equity": equity, "date": today.isoformat()})

    def validate(self, order: OrderRequest, state: MarketState) -> Optional[OrderRequest]:
        self.reset_day_if_needed(state.equity)
        if self.kill_switch_active:
            logger.warning("Kill switch active; blocking order", extra={"symbol": order.symbol})
            return None
        if self.relaxed_drawdown is False and self._breached_drawdown(state.equity):
            self._activate_kill_switch(reason="daily_drawdown")
            return None
        if len(state.open_positions) >= self.config.max_open_positions:
            logger.info("Max open positions reached", extra={"symbol": order.symbol})
            return None
        if order.leverage > self.config.max_leverage:
            logger.warning("Leverage too high", extra={"requested": order.leverage})
            return None
        if order.stop_loss is None:
            logger.warning("Stop loss required for sizing", extra={"symbol": order.symbol})
            return None

        symbol_info = self._resolve_symbol(order.symbol)
        reference_price = order.price or (state.candles[-1].close if state.candles else None)
        if reference_price is None:
            logger.warning("Cannot size order; missing price context", extra={"symbol": order.symbol})
            return None
        qty = self._size_position(state.equity, reference_price, order.stop_loss, symbol_info)
        if qty <= 0:
            return None

        qty = self._apply_notional_limits(qty, reference_price, order.symbol, state, symbol_info)
        if qty <= 0:
            return None

        rounded_price = symbol_info.round_price(reference_price) if symbol_info and reference_price else reference_price
        safe_order = order.model_copy(update={"quantity": qty, "price": rounded_price})
        logger.info(
            "Order approved by risk",
            extra={"symbol": safe_order.symbol, "qty": safe_order.quantity, "price": safe_order.price},
        )
        return safe_order

    def record_fill(self, pnl: float, *, commission_pct_of_equity: float = 0.0, is_closed_trade: bool = True) -> None:
        if self._daily_state is None:
            return
        if is_closed_trade:
            self._daily_state.realized_pnl += pnl
            self._daily_state.trades_count += 1
            if pnl < 0:
                self._daily_state.consecutive_losses += 1
            elif pnl > 0:
                self._daily_state.consecutive_losses = 0
        self._daily_state.commission_paid_pct += commission_pct_of_equity
        self._update_kill_switch()

    def _size_position(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        symbol_info: SymbolInfo | None,
    ) -> float:
        risk_amount = equity * self.config.max_risk_per_trade_pct
        distance = abs(entry_price - stop_price)
        if distance <= 0:
            logger.warning("Invalid stop distance", extra={"entry": entry_price, "stop": stop_price})
            return 0.0
        qty = risk_amount / distance
        if symbol_info:
            qty = symbol_info.round_qty(qty)
            if not symbol_info.validate_notional(entry_price, qty):
                min_qty = float(symbol_info.min_notional) / entry_price if entry_price else 0.0
                qty = symbol_info.round_qty(min_qty)
        return max(qty, 0.0)

    def _apply_notional_limits(
        self,
        qty: float,
        price: float,
        symbol: str,
        state: MarketState,
        symbol_info: SymbolInfo | None,
    ) -> float:
        notional = qty * price
        if notional < self.config.min_order_notional_usd:
            logger.info("Order below min notional", extra={"symbol": symbol, "notional": notional})
            return 0.0

        symbol_open_notional = sum(
            abs(pos.quantity * pos.entry_price)
            for pos in state.open_positions
            if pos.symbol == symbol and pos.is_open()
        )
        total_notional = sum(abs(pos.quantity * pos.entry_price) for pos in state.open_positions if pos.is_open())

        if self.config.max_symbol_notional_usd and notional + symbol_open_notional > self.config.max_symbol_notional_usd:
            allowed = max(self.config.max_symbol_notional_usd - symbol_open_notional, 0.0)
            qty = allowed / price if price else 0.0
        if self.config.max_total_notional_usd and notional + total_notional > self.config.max_total_notional_usd:
            allowed = max(self.config.max_total_notional_usd - total_notional, 0.0)
            qty = min(qty, allowed / price if price else 0.0)
        if symbol_info:
            qty = symbol_info.round_qty(qty)
        if qty * price < self.config.min_order_notional_usd:
            return 0.0
        return max(qty, 0.0)

    def _update_kill_switch(self) -> None:
        if self._daily_state is None or self.relaxed_drawdown:
            return
        reasons = []
        drawdown_limit = self.config.max_daily_drawdown_pct * 100.0
        if drawdown_limit > 0 and self._daily_state.drawdown_pct >= drawdown_limit:
            reasons.append("daily_drawdown")
        if (
            self.config.max_consecutive_losses > 0
            and self._daily_state.consecutive_losses >= self.config.max_consecutive_losses
        ):
            reasons.append("consecutive_losses")
        if (
            self.config.max_trades_per_day > 0
            and self._daily_state.trades_count >= self.config.max_trades_per_day
        ):
            reasons.append("trade_limit")
        if (
            self.config.max_commission_pct_per_day > 0
            and self._daily_state.commission_paid_pct >= self.config.max_commission_pct_per_day
        ):
            reasons.append("commission")
        if reasons:
            self._activate_kill_switch(reason="|".join(reasons))

    def _activate_kill_switch(self, reason: str) -> None:
        if self.kill_switch_active:
            return
        self.kill_switch_active = True
        log_event(
            "KILL_SWITCH_TRIGGERED",
            reason=reason,
            drawdown_pct=self._daily_state.drawdown_pct if self._daily_state else None,
            trades=self._daily_state.trades_count if self._daily_state else None,
            consecutive_losses=self._daily_state.consecutive_losses if self._daily_state else None,
        )
        logger.error("Kill switch activated", extra={"reason": reason})

    def _breached_drawdown(self, equity: float) -> bool:
        if self.relaxed_drawdown or self._daily_state is None:
            return False
        start_equity = self._daily_state.starting_equity
        if start_equity <= 0:
            return False
        dd_ratio = (start_equity - equity) / start_equity
        return dd_ratio >= self.config.max_daily_drawdown_pct

    def _resolve_symbol(self, symbol: str) -> SymbolInfo | None:
        if not self.symbol_resolver:
            return None
        try:
            return self.symbol_resolver.get(symbol)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Symbol resolver missing data", extra={"symbol": symbol, "error": str(exc)})
            return None
