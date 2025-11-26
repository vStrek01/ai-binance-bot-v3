from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from bot.core.config import BotConfig
from bot.exchange_info import SymbolFilters
from bot.execution.balance_manager import BalanceManager
from bot.execution.exchange_client import ExchangeClient, ExchangeRequestError
from bot.live_logging import OrderAuditLogger
from bot.utils.logger import get_logger
from infra.logging import log_event

logger = get_logger(__name__)


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    reduce_only: bool = False
    price: float | None = None
    time_in_force: str | None = None
    tag: str | None = None
    filters: SymbolFilters | None = None


@dataclass(slots=True)
class PlacedOrder:
    symbol: str
    side: str
    quantity: float
    price: float | None
    order_id: Optional[int]
    client_order_id: str
    reduce_only: bool
    raw: Dict[str, Any]
    duplicate: bool = False


@dataclass(slots=True)
class OrderError:
    category: str
    code: Optional[int]
    message: str
    retryable: bool


class OrderPlacementError(RuntimeError):
    def __init__(self, error: OrderError) -> None:
        super().__init__(f"Order failed ({error.category}): {error.message}")
        self.error = error


class OrderManager:
    """Centralizes live order submission with idempotency guarantees."""

    def __init__(
        self,
        cfg: BotConfig,
        client: ExchangeClient,
        logger: OrderAuditLogger,
        *,
        balance_manager: BalanceManager,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._cfg = cfg
        self._client = client
        self._logger = logger
        self._balance = balance_manager
        self._clock = clock or time.time
        self._cache_ttl = max(cfg.runtime.poll_interval_seconds, 10)
        self._bucket_size = max(int(cfg.runtime.poll_interval_seconds // 2) or 1, 1)
        self._id_sequence: Dict[int, int] = {}
        self._cache: Dict[Tuple[Any, ...], Tuple[PlacedOrder, float]] = {}

    def submit_order(self, request: OrderRequest) -> PlacedOrder:
        normalized_qty = self._normalize_quantity(request)
        if normalized_qty <= 0:
            raise OrderPlacementError(OrderError("validation", None, "Quantity must be positive", False))
        cache_key = self._fingerprint(request, normalized_qty)
        cached = self._cache.get(cache_key)
        now = self._clock()
        if cached and (now - cached[1]) < self._cache_ttl:
            cached_order = cached[0]
            return PlacedOrder(
                symbol=cached_order.symbol,
                side=cached_order.side,
                quantity=cached_order.quantity,
                price=cached_order.price,
                order_id=cached_order.order_id,
                client_order_id=cached_order.client_order_id,
                reduce_only=cached_order.reduce_only,
                raw=cached_order.raw,
                duplicate=True,
            )
        payload = self._build_payload(request, normalized_qty)
        attempts = 2 if not request.reduce_only else 1
        for attempt in range(attempts):
            self._logger.log({"event": "request", **payload})
            self._client.check_time_drift()
            self._emit_order_event(
                "ORDER_PLACED",
                request,
                quantity=self._parse_float(payload.get("quantity")) or normalized_qty,
                payload=payload,
                attempt=attempt + 1,
            )
            try:
                response = self._client.place_order(**payload)
            except ExchangeRequestError as exc:
                error = self._to_order_error(exc)
                self._logger.log(
                    {
                        "event": "error",
                        "symbol": request.symbol,
                        "side": request.side,
                        "category": error.category,
                        "message": error.message,
                        "code": error.code,
                    }
                )
                if error.category == "margin" and not request.reduce_only and attempt == 0:
                    normalized_qty = self._reduce_quantity(normalized_qty, request.filters)
                    if normalized_qty <= 0:
                        raise OrderPlacementError(error)
                    payload["quantity"] = self._format_quantity(normalized_qty)
                    continue
                raise OrderPlacementError(error) from exc
            order = self._record_success(cache_key, request, normalized_qty, response, now)
            self._emit_order_event(
                "ORDER_FILLED",
                request,
                quantity=order.quantity,
                payload=payload,
                response=response,
            )
            return order
        raise OrderPlacementError(OrderError("unknown", None, "Exceeded order retries", False))

    def _build_payload(self, request: OrderRequest, quantity: float) -> Dict[str, Any]:
        client_order_id = self._next_client_id(request.symbol, request.side)
        payload = {
            "symbol": request.symbol,
            "side": request.side,
            "type": request.order_type,
            "quantity": self._format_quantity(quantity),
            "newClientOrderId": client_order_id,
        }
        if request.reduce_only:
            payload["reduceOnly"] = "true"
        if request.price is not None and request.order_type != "MARKET":
            payload["price"] = self._format_price(request.price)
        if request.time_in_force:
            payload["timeInForce"] = request.time_in_force
        if request.tag:
            payload["newClientOrderId"] = f"{client_order_id}-{request.tag}"
        return payload

    def _record_success(
        self,
        cache_key: Tuple[Any, ...],
        request: OrderRequest,
        quantity: float,
        response: Dict[str, Any],
        timestamp: float,
    ) -> PlacedOrder:
        price = self._parse_float(response.get("avgPrice") or response.get("price"))
        if price is None:
            price = request.price
        order = PlacedOrder(
            symbol=request.symbol,
            side=request.side,
            quantity=self._parse_float(response.get("executedQty")) or quantity,
            price=price,
            order_id=self._parse_int(response.get("orderId")),
            client_order_id=str(response.get("clientOrderId") or response.get("newClientOrderId") or ""),
            reduce_only=request.reduce_only,
            raw=response,
        )
        self._logger.log({"event": "response", **response})
        self._cache[cache_key] = (order, timestamp)
        return order

    def _fingerprint(self, request: OrderRequest, qty: float) -> Tuple[Any, ...]:
        return (
            request.symbol,
            request.side,
            round(qty, 8),
            request.reduce_only,
            request.order_type,
            round(request.price or 0.0, 6),
        )

    def _next_client_id(self, symbol: str, side: str) -> str:
        bucket = int(self._clock() // self._bucket_size)
        seq = self._id_sequence.get(bucket, 0) + 1
        self._id_sequence[bucket] = seq
        return f"{symbol}-{side}-{bucket}-{seq}"

    def _normalize_quantity(self, request: OrderRequest) -> float:
        qty = max(request.quantity, 0.0)
        filters = request.filters
        if request.reduce_only:
            qty = self._balance.resolve_reduce_only_quantity(request.symbol, request.side, qty, request.price, filters)
        elif filters:
            qty = filters.adjust_quantity(qty)
        return qty

    def _reduce_quantity(self, quantity: float, filters: SymbolFilters | None) -> float:
        reduced = quantity * 0.5
        if filters:
            reduced = filters.adjust_quantity(reduced)
            if reduced < filters.min_qty:
                return 0.0
        return reduced

    def _format_quantity(self, quantity: float) -> str:
        return f"{max(quantity, 0.0):.8f}"

    def _format_price(self, price: float) -> str:
        return f"{max(price, 0.0):.8f}"

    def _to_order_error(self, exc: ExchangeRequestError) -> OrderError:
        retryable = exc.retryable
        return OrderError(category=exc.category, code=exc.code, message=str(exc), retryable=retryable)

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed

    @staticmethod
    def _parse_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _emit_order_event(
        self,
        event: str,
        request: OrderRequest,
        *,
        quantity: float,
        payload: Dict[str, Any],
        response: Dict[str, Any] | None = None,
        attempt: int | None = None,
    ) -> None:
        est_price = self._parse_float((response or {}).get("avgPrice"))
        if est_price is None:
            est_price = self._parse_float(payload.get("price"))
        if est_price is None and request.price is not None:
            est_price = float(request.price)
        size_usd = quantity * est_price if est_price else None
        log_event(
            event,
            run_mode=self._cfg.run_mode,
            symbol=request.symbol,
            side=request.side,
            qty=quantity,
            size_usd=size_usd,
            reduce_only=request.reduce_only,
            client_order_id=payload.get("newClientOrderId"),
            status=(response or {}).get("status"),
            attempt=attempt,
        )


__all__ = [
    "OrderManager",
    "OrderRequest",
    "PlacedOrder",
    "OrderPlacementError",
    "OrderError",
]
