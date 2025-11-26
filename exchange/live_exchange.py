from __future__ import annotations

from typing import List, Optional

from core.models import OrderFill, OrderRequest, Position
from exchange.base import Exchange
from exchange.binance_client import BinanceClient
from exchange.order_router import OrderRouter
from core.state import PositionManager


class LiveExchange(Exchange):
    """Exchange wrapper that routes orders through Binance for live trading."""

    def __init__(self, order_router: OrderRouter, client: BinanceClient, positions: PositionManager):
        self._order_router = order_router
        self._client = client
        self._positions = positions

    async def place_order(self, order: OrderRequest) -> Optional[OrderFill]:
        return await self._order_router.execute(order)

    async def cancel_order(self, symbol: str, client_order_id: str) -> bool:
        try:
            self._client.cancel_order(symbol, client_order_id)
            return True
        except Exception:  # pragma: no cover - network
            return False

    def get_balance(self) -> float:
        return self._positions.equity

    def get_open_positions(self) -> List[Position]:
        return self._positions.get_open_positions()

    def get_open_orders(self) -> List[dict]:
        # Live system relies on Binance UI / streaming for order state; return empty placeholder for now.
        return []
