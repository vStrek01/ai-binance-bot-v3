"""Resilient wrapper around Binance UMFutures with guarded retries."""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional

from binance.error import ClientError  # type: ignore[import-untyped]
try:  # pragma: no cover - optional import provided by python-binance
    from binance.exceptions import BinanceAPIException  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001 - fallback when module layout differs
    BinanceAPIException = Exception  # type: ignore[misc,assignment]
from binance.um_futures import UMFutures  # type: ignore[import-untyped]

try:  # pragma: no cover - requests is an optional runtime dep for unit tests
    from requests import exceptions as requests_exceptions
except Exception:  # noqa: BLE001 - fallback when requests is unavailable
    class _RequestsFallback:  # type: ignore[too-many-ancestors]
        RequestException = Timeout = ConnectionError = Exception

    requests_exceptions = _RequestsFallback()

from bot.utils.logger import get_logger
from infra.alerts import send_alert
from infra.logging import log_event

logger = get_logger(__name__)


class ExchangeRequestError(RuntimeError):
    """Raised when an exchange request ultimately fails after retries."""

    def __init__(
        self,
        operation: str,
        category: str,
        message: str,
        *,
        code: Optional[int] = None,
        status_code: Optional[int] = None,
        original: Exception | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(f"{operation} failed ({category}): {message}")
        self.operation = operation
        self.category = category
        self.code = code
        self.status_code = status_code
        self.original = original
        self.retryable = retryable


class ExchangeClient:
    """Thin wrapper over UMFutures that enforces retries and classification."""

    def __init__(
        self,
        client: UMFutures,
        *,
        mode: str,
        max_retries: int = 3,
        backoff: float = 0.5,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown: float = 30.0,
    ) -> None:
        self._client: Any = client
        self.mode = mode
        self._max_retries = max(1, max_retries)
        self._base_backoff = max(backoff, 0.1)
        self._max_backoff = self._base_backoff * 4
        self._consecutive_failures = 0
        self._circuit_threshold = max(circuit_breaker_threshold, 1)
        self._circuit_cooldown = max(circuit_breaker_cooldown, 1.0)
        self._circuit_open_until = 0.0
        self._last_drift_check = 0.0

    @property
    def raw(self) -> UMFutures:
        return self._client

    def exchange_info(self) -> Dict[str, Any]:
        return self._call("exchange_info", self._client.exchange_info)

    def leverage_bracket(self) -> List[Dict[str, Any]]:
        return self._call("leverage_bracket", self._client.leverage_bracket)

    def get_klines(self, **params: Any) -> List[List[Any]]:
        return self._call("klines", self._client.klines, **params)

    def get_balance(self) -> List[Dict[str, Any]]:
        return self._call("balance", self._client.balance)

    def change_leverage(self, **params: Any) -> Dict[str, Any]:
        return self._call("change_leverage", self._client.change_leverage, **params)

    def place_order(self, **params: Any) -> Dict[str, Any]:
        return self._call("new_order", self._client.new_order, **params)

    def get_position_risk(self) -> List[Dict[str, Any]]:
        return self._call("get_position_risk", self._client.get_position_risk)

    def get_account_trades(self, **params: Any) -> List[Dict[str, Any]]:
        return self._call("get_account_trades", self._client.get_account_trades, **params)

    def cancel_order(self, **params: Any) -> Dict[str, Any]:  # pragma: no cover - future live usage
        return self._call("cancel_order", self._client.cancel_order, **params)

    def cancel_all_orders(self, **params: Any) -> Dict[str, Any]:  # pragma: no cover - future live usage
        return self._call("cancel_all_open_orders", self._client.cancel_all_open_orders, **params)

    def check_time_drift(self, warn_threshold: float = 2.0, abort_threshold: float = 5.0, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_drift_check) < 30.0:
            return
        try:
            payload = self._client.time()
        except Exception as exc:  # noqa: BLE001 - network/SDK issues
            logger.warning("Server time check failed", extra={"error": str(exc)})
            return
        checked_at = time.time()
        self._last_drift_check = checked_at
        server_ms = int((payload or {}).get("serverTime", 0))
        if server_ms <= 0:
            return
        local_ms = int(checked_at * 1000)
        drift = abs(server_ms - local_ms) / 1000.0
        if drift >= abort_threshold:
            log_event("TIME_DRIFT_CRITICAL", drift_seconds=drift)
            raise ExchangeRequestError("server_time", "drift", f"Clock drift {drift:.2f}s exceeds abort threshold")
        if drift >= warn_threshold:
            log_event("TIME_DRIFT_WARNING", drift_seconds=drift)

    def _call(self, label: str, func: Callable[..., Any], **params: Any) -> Any:
        self._enforce_circuit(label)
        delay = self._base_backoff
        last_error: Optional[ExchangeRequestError] = None
        for attempt in range(self._max_retries):
            try:
                result = func(**params)
            except ClientError as exc:  # noqa: BLE001
                category = self._classify_client_error(exc)
                if self._should_retry(category) and attempt + 1 < self._max_retries:
                    self._sleep_with_backoff(label, category, attempt + 1, delay)
                    delay = min(delay * 2, self._max_backoff)
                    continue
                last_error = ExchangeRequestError(
                    operation=label,
                    category=category,
                    message=str(exc),
                    code=getattr(exc, "error_code", None),
                    status_code=getattr(exc, "status_code", None),
                    original=exc,
                    retryable=self._should_retry(category),
                )
                self._record_failure(category)
                break
            except BinanceAPIException as exc:  # type: ignore[misc]
                category = self._classify_binance_exception(exc)
                if self._should_retry(category) and attempt + 1 < self._max_retries:
                    self._sleep_with_backoff(label, category, attempt + 1, delay)
                    delay = min(delay * 2, self._max_backoff)
                    continue
                last_error = ExchangeRequestError(
                    operation=label,
                    category=category,
                    message=str(exc),
                    code=getattr(exc, "code", None),
                    status_code=getattr(exc, "status_code", None),
                    original=exc,
                    retryable=self._should_retry(category),
                )
                self._record_failure(category)
                break
            except (requests_exceptions.Timeout, requests_exceptions.ConnectionError, requests_exceptions.RequestException, OSError) as exc:  # type: ignore[attr-defined]
                category = "network"
                if self._should_retry(category) and attempt + 1 < self._max_retries:
                    self._sleep_with_backoff(label, category, attempt + 1, delay)
                    delay = min(delay * 2, self._max_backoff)
                    continue
                last_error = ExchangeRequestError(
                    operation=label,
                    category=category,
                    message=str(exc),
                    original=exc,
                    retryable=self._should_retry(category),
                )
                self._record_failure(category)
                break
            except Exception as exc:  # pragma: no cover - unexpected
                category = "unknown"
                last_error = ExchangeRequestError(operation=label, category=category, message=str(exc), original=exc)
                self._record_failure(category)
                break
            else:
                self._record_success()
                return result
        if last_error is not None:
            raise last_error
        raise ExchangeRequestError(operation=label, category="unknown", message="Exceeded retry budget")

    @staticmethod
    def _should_retry(category: str) -> bool:
        return category in {"rate_limit", "server", "network"}

    @staticmethod
    def _classify_client_error(exc: ClientError) -> str:
        code = getattr(exc, "error_code", None)
        status = getattr(exc, "status_code", 0)
        if code == -1021:
            log_event("TIME_DRIFT_SERVER_REJECTION", code=code, status=status)
            return "drift"
        if code in {-2014, -2015} or status in {401, 403}:
            return "auth"
        if code in {-1003, -1015} or status == 429:
            return "rate_limit"
        if status and status >= 500:
            return "server"
        if code in {-2019, -2021, -2026}:
            return "margin"
        return "client"

    @staticmethod
    def _classify_binance_exception(exc: BinanceAPIException) -> str:  # type: ignore[misc]
        status = getattr(exc, "status_code", 0)
        code = getattr(exc, "code", None)
        if code == -1021:
            log_event("TIME_DRIFT_SERVER_REJECTION", code=code, status=status)
            return "drift"
        if status in {401, 403}:
            return "auth"
        if status == 429:
            return "rate_limit"
        if status >= 500:
            return "server"
        return "client"

    def _sleep_with_backoff(self, label: str, category: str, attempt: int, delay: float) -> None:
        jitter = random.uniform(0, delay * 0.1)
        wait_time = delay + jitter
        logger.warning("[%s] %s error; retry %s/%s in %.2fs", label, category, attempt, self._max_retries, wait_time)
        time.sleep(wait_time)

    def _record_failure(self, category: str) -> None:
        if category not in {"server", "network", "rate_limit"}:
            self._consecutive_failures = 0
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._circuit_threshold:
            self._circuit_open_until = time.time() + self._circuit_cooldown
            logger.error(
                "Exchange circuit opened for %.1fs after %s consecutive %s errors",
                self._circuit_cooldown,
                self._consecutive_failures,
                category,
            )
            payload = {
                "cooldown": self._circuit_cooldown,
                "failures": self._consecutive_failures,
                "category": category,
            }
            log_event("exchange_circuit_open", **payload)
            send_alert(
                "EXCHANGE_CIRCUIT_OPEN",
                severity="critical",
                message="Exchange circuit breaker engaged",
                **payload,
            )

    def _record_success(self) -> None:
        if self._consecutive_failures > 0 or self._circuit_open_until > 0:
            logger.info("Exchange circuit recovered after %s consecutive failures", self._consecutive_failures)
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _enforce_circuit(self, label: str) -> None:
        now = time.time()
        if now < self._circuit_open_until:
            remaining = self._circuit_open_until - now
            raise ExchangeRequestError(
                operation=label,
                category="circuit_open",
                message=f"circuit breaker open for {remaining:.1f}s",
            )


__all__ = ["ExchangeClient", "ExchangeRequestError"]
