from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.models import MarketState, OrderRequest, RiskConfig
from core.safety import SafetyLimits, cap_notional
from infra.logging import logger


class RiskManager:
    def __init__(self, config: RiskConfig, *, relaxed_drawdown: bool = False, safety_limits: Optional[SafetyLimits] = None):
        self.config = config
        self.daily_start_equity: Optional[float] = None
        self.last_reset: Optional[datetime] = None
        self.relaxed_drawdown = relaxed_drawdown
        self._drawdown_alerted = False
        self.safety_limits = safety_limits

    def reset_day_if_needed(self, equity: float):
        now = datetime.utcnow()
        if self.last_reset is None:
            self.last_reset = now
            if self.daily_start_equity is None:
                self.daily_start_equity = equity
            logger.info("Daily risk counters reset", extra={"equity": equity})
            self._drawdown_alerted = False
            return
        if now.date() != self.last_reset.date():
            self.last_reset = now
            self.daily_start_equity = equity
            logger.info("Daily risk counters reset", extra={"equity": equity})
            self._drawdown_alerted = False

    def _daily_drawdown_exceeded(self, equity: float) -> bool:
        if self.relaxed_drawdown:
            return False
        if self.daily_start_equity is None:
            self.daily_start_equity = equity
            return False
        dd = (equity - self.daily_start_equity) / self.daily_start_equity
        return dd <= -self.config.max_daily_drawdown_pct

    def enable_backtest_mode(self) -> None:
        self.relaxed_drawdown = True

    def position_size(self, equity: float, entry_price: float, stop_loss: Optional[float]) -> float:
        if stop_loss is None:
            raise ValueError("stop_loss required for R-based sizing")
        risk_amount = equity * self.config.max_risk_per_trade_pct
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            raise ValueError("stop distance zero")
        qty = risk_amount / stop_distance
        return qty

    def validate(self, order: OrderRequest, state: MarketState) -> Optional[OrderRequest]:
        self.reset_day_if_needed(state.equity)

        if self._daily_drawdown_exceeded(state.equity):
            if not self._drawdown_alerted:
                logger.error("Daily drawdown limit hit", extra={"equity": state.equity})
                self._drawdown_alerted = True
            return None

        if len(state.open_positions) >= self.config.max_open_positions:
            logger.warning("Max open positions reached", extra={"count": len(state.open_positions)})
            return None

        if order.leverage > self.config.max_leverage:
            logger.warning("Leverage too high", extra={"requested": order.leverage})
            return None

        try:
            reference_price = order.price or (state.candles[-1].close if state.candles else None)
            if reference_price is None:
                logger.warning("Cannot size order: missing price context", extra={"symbol": state.symbol})
                return None
            qty = self.position_size(state.equity, reference_price, order.stop_loss)
        except ValueError as exc:
            logger.warning("Invalid risk parameters", extra={"error": str(exc)})
            return None

        qty = self._apply_notional_caps(qty, reference_price, order, state)
        if qty <= 0:
            return None

        if qty <= 0:
            logger.warning("Computed size non-positive", extra={"qty": qty})
            return None

        safe_order = order.model_copy(update={"quantity": qty})
        logger.info("Risk validated order", extra={"order": safe_order.model_dump()})
        return safe_order

    def _apply_notional_caps(self, qty: float, price: float, order: OrderRequest, state: MarketState) -> float:
        if self.safety_limits is None:
            return qty

        current_symbol_notional = sum(
            abs(pos.quantity * pos.entry_price) for pos in state.open_positions if pos.symbol == order.symbol and pos.is_open()
        )
        current_total_notional = sum(abs(pos.quantity * pos.entry_price) for pos in state.open_positions if pos.is_open())
        capped_qty = cap_notional(
            qty,
            price,
            max_symbol_notional=self.config.max_symbol_notional_usd,
            max_total_notional=self.safety_limits.max_total_notional_usd,
            current_symbol_notional=current_symbol_notional,
            current_total_notional=current_total_notional,
        )
        if capped_qty <= 0:
            logger.warning(
                "Order blocked by notional caps",
                extra={
                    "symbol": order.symbol,
                    "requested_notional": qty * price,
                    "symbol_notional": current_symbol_notional,
                    "total_notional": current_total_notional,
                },
            )
            return 0.0

        notional_after_cap = capped_qty * price
        if notional_after_cap < self.config.min_order_notional_usd:
            logger.info(
                "SKIP_TRADE_TOO_SMALL_AFTER_CAP",
                extra={
                    "symbol": order.symbol,
                    "notional": notional_after_cap,
                    "min_notional": self.config.min_order_notional_usd,
                },
            )
            return 0.0

        if capped_qty < qty:
            logger.info(
                "Order size reduced by notional caps",
                extra={
                    "symbol": order.symbol,
                    "original_qty": qty,
                    "capped_qty": capped_qty,
                    "price": price,
                },
            )
        return capped_qty
