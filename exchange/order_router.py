from __future__ import annotations

import hashlib
import time
from typing import Optional, Tuple

from core.models import OrderFill, OrderRequest, OrderType, Side
from exchange.binance_client import BinanceClient
from exchange.symbols import SymbolResolver
from infra.logging import logger, log_event


class OrderRouter:
    def __init__(self, client: BinanceClient, symbol_resolver: Optional[SymbolResolver] = None):
        self.client = client
        self._symbol_resolver = symbol_resolver

    def _client_order_id(self, order: OrderRequest) -> str:
        raw = f"{order.symbol}-{order.side}-{order.order_type}-{round(time.time() * 1000)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _normalize_price_components(
        self, order: OrderRequest
    ) -> Tuple[Optional[float], float, Optional[float], Optional[float]]:
        info = self._symbol_resolver.get(order.symbol) if self._symbol_resolver else None
        price = order.price
        qty = order.quantity
        stop_loss = order.stop_loss
        take_profit = order.take_profit
        raw_snapshot = {
            "price_before": price,
            "qty_before": qty,
            "stop_loss_before": stop_loss,
            "take_profit_before": take_profit,
        }
        if info:
            if price is not None:
                price = info.round_price(price)
            qty = info.round_qty(qty)
            if stop_loss is not None:
                stop_loss = info.round_price(stop_loss)
            if take_profit is not None:
                take_profit = info.round_price(take_profit)
            log_event(
                "exchange_filters_applied",
                symbol=order.symbol,
                tick_size=float(info.tick_size),
                step_size=float(info.step_size),
                min_qty=float(info.min_qty),
                min_notional=float(info.min_notional),
                price_after=price,
                qty_after=qty,
                stop_loss_after=stop_loss,
                take_profit_after=take_profit,
                **raw_snapshot,
            )
        if qty <= 0:
            raise ValueError("Normalized quantity must be positive")
        if info and price is not None and not info.validate_notional(price, qty):
            raise ValueError("Order notional below symbol minimum")
        return price, qty, stop_loss, take_profit

    async def execute(self, order: OrderRequest) -> Optional[OrderFill]:
        try:
            self.client.check_time_drift()
        except RuntimeError as exc:
            log_event("TIME_DRIFT_ABORTED_ORDER", symbol=order.symbol, error=str(exc))
            logger.error("Aborting order due to clock drift", extra={"error": str(exc)})
            return None

        try:
            normalized_price, normalized_qty, normalized_sl, normalized_tp = self._normalize_price_components(order)
        except ValueError as exc:
            logger.warning("Order rejected by symbol filters", extra={"symbol": order.symbol, "error": str(exc)})
            log_event("ORDER_FILTER_BLOCKED", symbol=order.symbol, error=str(exc))
            return None

        params = {
            "symbol": order.symbol,
            "side": "BUY" if order.side == Side.LONG else "SELL",
            "type": order.order_type.value,
            "quantity": round(normalized_qty, 6),
            "newClientOrderId": self._client_order_id(order),
        }
        if order.order_type == OrderType.LIMIT and normalized_price:
            params["price"] = normalized_price
            params["timeInForce"] = "GTC"

        if order.reduce_only:
            params["reduceOnly"] = True

        try:
            est_price = normalized_price or params.get("price")
            size_usd = abs(normalized_qty * est_price) if est_price else None
            log_event(
                "ORDER_PLACED",
                symbol=order.symbol,
                side=order.side.value,
                qty=normalized_qty,
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
            sl = normalized_sl or order.stop_loss
            tp = normalized_tp or order.take_profit
            if sl and tp:
                self._create_oco(order, params["newClientOrderId"], stop_loss=sl, take_profit=tp)
        except Exception as exc:  # pragma: no cover - network
            logger.error("Failed to create OCO", extra={"error": str(exc)})

        return fill

    def _create_oco(self, order: OrderRequest, client_order_id: str, *, stop_loss: float, take_profit: float):
        side = "SELL" if order.side == Side.LONG else "BUY"
        stop_params = {
            "symbol": order.symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": stop_loss,
            "closePosition": True,
            "newClientOrderId": f"{client_order_id}-sl",
        }
        tp_params = {
            "symbol": order.symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": take_profit,
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
