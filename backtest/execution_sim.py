from __future__ import annotations

from datetime import datetime
from typing import Tuple

from core.models import OrderFill, OrderRequest, Side


class ExecutionSimulator:
    def __init__(self, spread: float, slippage: float, fee_rate: float):
        self.spread = spread
        self.slippage = slippage
        self.fee_rate = fee_rate

    def simulate(self, order: OrderRequest, price: float) -> Tuple[OrderFill, float]:
        adj_price = price * (1 + self.slippage if order.side == Side.LONG else 1 - self.slippage)
        if order.side == Side.LONG:
            adj_price += self.spread
        else:
            adj_price -= self.spread
        fee = abs(adj_price * order.quantity) * self.fee_rate
        fill = OrderFill(
            order_id="sim", status="FILLED", filled_qty=order.quantity, avg_price=adj_price, timestamp=datetime.utcnow()
        )
        return fill, fee
