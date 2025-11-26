from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, List, Optional

import websockets

from core.models import Candle
from infra.logging import logger


class BinanceStream:
    def __init__(self, symbol: str, interval: str = "1m", testnet: bool = True, ws_market_url: Optional[str] = None):
        self.symbol = symbol.upper()
        self.interval = interval
        self.testnet = testnet
        base_url = ws_market_url or ("wss://fstream.binancefuture.com" if testnet else "wss://fstream.binance.com")
        self.ws_url = self._build_stream_url(base_url)
        self.history: List[Candle] = []

    def _build_stream_url(self, base_url: str) -> str:
        normalized = (base_url or "").strip()
        if not normalized:
            normalized = "wss://fstream.binancefuture.com" if self.testnet else "wss://fstream.binance.com"
        stream_suffix = f"{self.symbol.lower()}@kline_{self.interval}"
        if "{symbol}" in normalized or "{interval}" in normalized:
            return normalized.format(symbol=self.symbol.lower(), interval=self.interval)
        if "streams=" in normalized:
            if "{" in normalized:
                return normalized.format(symbol=self.symbol.lower(), interval=self.interval)
            if normalized.endswith("="):
                return f"{normalized}{stream_suffix}"
            return normalized
        normalized = normalized.rstrip("/")
        endpoint = f"{normalized}/stream"
        separator = "&" if "?" in endpoint else "?"
        return f"{endpoint}{separator}streams={stream_suffix}"

    async def candle_stream(self) -> AsyncGenerator[Optional[Candle], None]:
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=10) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        kline = data.get("data", {}).get("k")
                        if not kline or not kline.get("x"):
                            continue
                        candle = Candle(
                            symbol=self.symbol,
                            open_time=datetime.fromtimestamp(kline["t"] / 1000),
                            close_time=datetime.fromtimestamp(kline["T"] / 1000),
                            open=float(kline["o"]),
                            high=float(kline["h"]),
                            low=float(kline["l"]),
                            close=float(kline["c"]),
                            volume=float(kline["v"]),
                        )
                        self.history.append(candle)
                        yield candle
            except Exception as exc:  # pragma: no cover - network
                logger.error("Websocket error", extra={"error": str(exc)})
                await asyncio.sleep(5)
                continue
