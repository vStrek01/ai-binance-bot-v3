from __future__ import annotations

import hashlib
import time
from typing import Optional

from core.models import OrderFill, OrderRequest, OrderType, Side
from exchange.binance_client import BinanceClient
from infra.logging import logger, log_event


class OrderRouter:
    def __init__(self, client: BinanceClient):
        self.client = client

    def _client_order_id(self, order: OrderRequest) -> str:
        raw = f"{order.symbol}-{order.side}-{order.order_type}-{round(time.time() * 1000)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    async def execute(self, order: OrderRequest) -> Optional[OrderFill]:
        try:
            self.client.check_time_drift()
        except RuntimeError as exc:
            log_event("TIME_DRIFT_ABORTED_ORDER", symbol=order.symbol, error=str(exc))
            logger.error("Aborting order due to clock drift", extra={"error": str(exc)})
            return None

        params = {
            "symbol": order.symbol,
            "side": "BUY" if order.side == Side.LONG else "SELL",
            "type": order.order_type.value,
            "quantity": round(order.quantity, 6),
            "newClientOrderId": self._client_order_id(order),
        }
        if order.order_type == OrderType.LIMIT and order.price:
            params["price"] = order.price
            params["timeInForce"] = "GTC"

        if order.reduce_only:
            params["reduceOnly"] = True

        try:
            est_price = order.price or params.get("price")
            size_usd = abs(order.quantity * est_price) if est_price else None
            log_event(
                "ORDER_PLACED",
                symbol=order.symbol,
                side=order.side.value,
                qty=order.quantity,
                size_usd=size_usd,
                entry_price=est_price,
                leverage=order.leverage,
            )
            response = self.client.place_order(params)
            logger.info("Order placed", extra={"response": response})
        except Exception as exc:  # pragma: no cover - network
            logger.error("Order placement failed", extra={"error": str(exc)})
            log_event("ORDER_FAILED", symbol=order.symbol, error=str(exc))
            return None

        fill_price = float(response.get("avgPrice") or response.get("price") or params.get("price", 0))
        filled_qty = float(response.get("executedQty", order.quantity))

        fill = OrderFill(
            order_id=str(response.get("orderId")),
            status=response.get("status", "NEW"),
            filled_qty=filled_qty,
            avg_price=fill_price,
            timestamp=datetime_from_ms(response.get("updateTime") or response.get("transactTime")),
            client_order_id=params["newClientOrderId"],
        )
        log_event(
            "ORDER_FILLED",
            symbol=order.symbol,
            side=order.side.value,
            fill_price=fill.avg_price,
            qty=fill.filled_qty,
            status=fill.status,
            client_order_id=fill.client_order_id,
        )

        # create protection orders
        try:
            if order.stop_loss and order.take_profit:
                self._create_oco(order, params["newClientOrderId"])
        except Exception as exc:  # pragma: no cover - network
            logger.error("Failed to create OCO", extra={"error": str(exc)})

        return fill

    def _create_oco(self, order: OrderRequest, client_order_id: str):
        side = "SELL" if order.side == Side.LONG else "BUY"
        stop_params = {
            "symbol": order.symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": order.stop_loss,
            "closePosition": True,
            "newClientOrderId": f"{client_order_id}-sl",
        }
        tp_params = {
            "symbol": order.symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": order.take_profit,
            "closePosition": True,
            "newClientOrderId": f"{client_order_id}-tp",
        }
        self.client.place_order(stop_params)
        self.client.place_order(tp_params)
        logger.info("Placed protective orders", extra={"stop": stop_params, "tp": tp_params})


def datetime_from_ms(ts: Optional[int]):
    from datetime import datetime

    if ts is None:
        return datetime.utcnow()
    return datetime.fromtimestamp(ts / 1000)
