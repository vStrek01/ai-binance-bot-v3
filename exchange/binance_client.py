from __future__ import annotations

import hashlib
import hmac
import os
import time
import random
from collections import OrderedDict
from typing import Any, Dict, Optional

import requests

from infra.logging import log_event, logger


class BinanceClient:
    MAINNET = "https://fapi.binance.com"
    TESTNET = "https://testnet.binancefuture.com"

    MAX_ATTEMPTS = 3
    BASE_BACKOFF = 0.5
    MAX_BACKOFF = 2.0

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: Optional[str] = None,
        testnet: bool = True,
        recv_window: int = 5000,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        fallback = self.TESTNET if testnet else self.MAINNET
        self.base_url = (base_url or fallback).rstrip("/")
        self.recv_window = recv_window

    def _timestamp(self) -> int:
        return int(time.time() * 1000)

    def _sign(self, params: "OrderedDict[str, Any]") -> str:
        query = "&".join([f"{key}={value}" for key, value in params.items()])
        return hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()

    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Any:
        url = f"{self.base_url}{path}"
        params = dict(params or {})

        last_error: Optional[str] = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                request_params = self._prepare_params(params, signed)
                resp = requests.request(method, url, params=request_params, headers=self._headers(), timeout=10)
                if resp.status_code >= 400:
                    self._log_account_diagnostic(resp, path, attempt)
                if resp.status_code == 418 and "recvWindow" in resp.text:
                    params["recvWindow"] = max(self.recv_window, 10_000)
                if resp.status_code == 429:
                    log_event("RATE_LIMIT_HIT", path=path, attempt=attempt, status=resp.status_code)
                    self._sleep_with_backoff(attempt)
                    continue
                if resp.status_code >= 500:
                    log_event("EXCHANGE_RETRY", path=path, attempt=attempt, status=resp.status_code)
                    self._sleep_with_backoff(attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = str(exc)
                logger.warning("Binance request failed", extra={"attempt": attempt, "error": last_error})
                log_event("EXCHANGE_RETRY", path=path, attempt=attempt, error=last_error)
                self._sleep_with_backoff(attempt)
        log_event("EXCHANGE_REQUEST_FAILED", path=path, error=last_error, attempts=self.MAX_ATTEMPTS)
        raise RuntimeError(f"Binance request failed after retries: {path}")

    def _prepare_params(self, params: Dict[str, Any], signed: bool) -> Any:
        if not signed:
            return params
        working_params: Dict[str, Any] = dict(params)
        working_params.setdefault("recvWindow", self.recv_window)
        working_params["timestamp"] = self._timestamp()
        ordered = OrderedDict((key, working_params[key]) for key in sorted(working_params.keys()) if working_params[key] is not None)
        signature = self._sign(ordered)
        ordered["signature"] = signature
        return ordered

    def _log_account_diagnostic(self, response: requests.Response, path: str, attempt: int) -> None:
        if path != "/fapi/v2/account":
            return
        body_preview = (response.text or "")[:500]
        log_event(
            "EXCHANGE_ACCOUNT_DIAGNOSTIC",
            path=path,
            attempt=attempt,
            status=response.status_code,
            body=body_preview,
        )

    def _sleep_with_backoff(self, attempt: int) -> None:
        delay = min(self.BASE_BACKOFF * (2 ** (attempt - 1)), self.MAX_BACKOFF)
        time.sleep(delay + random.uniform(0.0, 0.25))

    def check_time_drift(self, warn_threshold: float = 2.0, abort_threshold: float = 5.0) -> None:
        """Validate local clock against Binance server time and log drift."""

        try:
            payload = self._request("GET", "/fapi/v1/time")
        except RuntimeError as exc:  # pragma: no cover - network
            logger.warning("Unable to verify server time", extra={"error": str(exc)})
            return
        server_ms = int(payload.get("serverTime", 0))
        if not server_ms:
            return
        local_ms = self._timestamp()
        drift = abs(server_ms - local_ms) / 1000.0
        if drift >= abort_threshold:
            log_event("TIME_DRIFT_CRITICAL", drift_seconds=drift)
            raise RuntimeError(f"Local clock drift {drift:.2f}s exceeds abort threshold")
        if drift >= warn_threshold:
            log_event("TIME_DRIFT_WARNING", drift_seconds=drift)

    def place_order(self, params: Dict[str, Any]) -> Any:
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def cancel_order(self, symbol: str, orig_client_order_id: str) -> Any:
        params = {"symbol": symbol, "origClientOrderId": orig_client_order_id}
        return self._request("DELETE", "/fapi/v1/order", params=params, signed=True)

    def exchange_info(self) -> Any:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def get_exchange_info(self) -> Any:
        """Alias used by SymbolResolver for clarity."""

        return self.exchange_info()

    def ping(self) -> None:
        """Ensure the exchange is reachable."""

        self._request("GET", "/fapi/v1/ping")

    def server_time_ms(self) -> int:
        """Return the exchange server time in milliseconds."""

        payload = self._request("GET", "/fapi/v1/time")
        return int(payload.get("serverTime", 0))

    # --- Account helpers -------------------------------------------------

    def account_overview(self) -> Dict[str, Any]:
        """Return the futures account payload for balance/permission checks."""

        payload = self._request("GET", "/fapi/v2/account", signed=True)
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected account response")
        return payload

    def account_balances(self) -> list[Dict[str, Any]]:
        """Return the raw asset balances from the futures account payload."""

        account = self.account_overview()
        assets = account.get("assets")
        if isinstance(assets, list):
            return [asset for asset in assets if isinstance(asset, dict)]
        return []
