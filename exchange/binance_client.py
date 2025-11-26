from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, Optional

import requests

from infra.logging import logger


class BinanceClient:
    MAINNET = "https://fapi.binance.com"
    TESTNET = "https://testnet.binancefuture.com"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, recv_window: int = 5000):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = self.TESTNET if testnet else self.MAINNET
        self.recv_window = recv_window

    def _timestamp(self) -> int:
        return int(time.time() * 1000)

    def _sign(self, params: Dict[str, Any]) -> str:
        query = "&".join([f"{k}={params[k]}" for k in sorted(params.keys()) if params[k] is not None])
        return hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()

    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Any:
        url = f"{self.base_url}{path}"
        params = params or {}
        if signed:
            params.setdefault("timestamp", self._timestamp())
            params.setdefault("recvWindow", self.recv_window)
            params["signature"] = self._sign(params)

        for attempt in range(5):
            try:
                resp = requests.request(method, url, params=params, headers=self._headers(), timeout=10)
                if resp.status_code == 418 and "recvWindow" in resp.text:
                    params["recvWindow"] = max(self.recv_window, 10_000)
                if resp.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("Binance request failed", extra={"attempt": attempt, "error": str(exc)})
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Binance request failed after retries: {path}")

    def place_order(self, params: Dict[str, Any]) -> Any:
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def cancel_order(self, symbol: str, orig_client_order_id: str) -> Any:
        params = {"symbol": symbol, "origClientOrderId": orig_client_order_id}
        return self._request("DELETE", "/fapi/v1/order", params=params, signed=True)

    def exchange_info(self) -> Any:
        return self._request("GET", "/fapi/v1/exchangeInfo")
