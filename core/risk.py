from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.models import MarketState, OrderRequest, RiskConfig
from infra.logging import logger


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_start_equity: Optional[float] = None
        self.last_reset: Optional[datetime] = None

    def reset_day_if_needed(self, equity: float):
        now = datetime.utcnow()
        if self.last_reset is None:
            self.last_reset = now
            if self.daily_start_equity is None:
                self.daily_start_equity = equity
            logger.info("Daily risk counters reset", extra={"equity": equity})
            return
        if now.date() != self.last_reset.date():
            self.last_reset = now
            self.daily_start_equity = equity
            logger.info("Daily risk counters reset", extra={"equity": equity})

    def _daily_drawdown_exceeded(self, equity: float) -> bool:
        if self.daily_start_equity is None:
            self.daily_start_equity = equity
            return False
        dd = (equity - self.daily_start_equity) / self.daily_start_equity
        return dd <= -self.config.max_daily_drawdown_pct

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
            logger.error("Daily drawdown limit hit", extra={"equity": state.equity})
            return None

        if len(state.open_positions) >= self.config.max_open_positions:
            logger.warning("Max open positions reached", extra={"count": len(state.open_positions)})
            return None

        if order.leverage > self.config.max_leverage:
            logger.warning("Leverage too high", extra={"requested": order.leverage})
            return None

        try:
            qty = self.position_size(state.equity, order.price or state.candles[-1].close, order.stop_loss)
        except ValueError as exc:
            logger.warning("Invalid risk parameters", extra={"error": str(exc)})
            return None

        if qty <= 0:
            logger.warning("Computed size non-positive", extra={"qty": qty})
            return None

        safe_order = order.model_copy(update={"quantity": qty})
        logger.info("Risk validated order", extra={"order": safe_order.model_dump()})
        return safe_order
