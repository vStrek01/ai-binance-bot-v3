from __future__ import annotations

from typing import List, Optional

from core.models import Candle, OrderFill, OrderRequest, Position, Side
from core.state import PositionManager
from exchange.base import Exchange
from infra.logging import log_event


class SimulatedExchange(Exchange):
    """Simple exchange implementation for backtests, using candle data for fills."""

    def __init__(self, positions: PositionManager, *, spread: float, slippage: float):
        self._positions = positions
        self._spread = spread
        self._slippage = slippage
        self._current_candle: Optional[Candle] = None

    def update_market(self, candle: Candle) -> None:
        self._current_candle = candle

    async def place_order(self, order: OrderRequest) -> Optional[OrderFill]:
        if self._current_candle is None:
            raise RuntimeError("Market data not initialized for simulated exchange")
        fill_price = self._fill_price(order.side, self._current_candle.close)
        log_event(
            "ORDER_PLACED",
            symbol=order.symbol,
            side=order.side.value,
            qty=order.quantity,
            entry_price=fill_price,
            run_mode=self._positions.run_mode,
        )
        fill = OrderFill(
            order_id="sim",
            status="FILLED",
            filled_qty=order.quantity,
            avg_price=fill_price,
            timestamp=self._current_candle.close_time,
            client_order_id=None,
        )
        self._positions.update_on_fill(
            fill,
            side=order.side,
            symbol=order.symbol,
            leverage=order.leverage,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
        )
        log_event(
            "ORDER_FILLED",
            symbol=order.symbol,
            side=order.side.value,
            fill_price=fill.avg_price,
            qty=fill.filled_qty,
            status=fill.status,
            run_mode=self._positions.run_mode,
        )
        return fill

    async def cancel_order(self, symbol: str, client_order_id: str) -> bool:
        return True

    def get_balance(self) -> float:
        return self._positions.equity

    def get_open_positions(self) -> List[Position]:
        return self._positions.get_open_positions()

    def get_open_orders(self) -> List[dict[str, str]]:
        return []

    def _fill_price(self, side: Side, mid_price: float) -> float:
        slip = mid_price * self._slippage
        if side == Side.LONG:
            return mid_price + self._spread + slip
        return mid_price - self._spread - slip
