from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, List, Optional

import websockets

from core.models import Candle
from infra.logging import logger


class BinanceStream:
    def __init__(self, symbol: str, interval: str = "1m", testnet: bool = True):
        self.symbol = symbol.upper()
        self.interval = interval
        self.testnet = testnet
        self.ws_url = (
            f"wss://fstream.binance.com/stream?streams={self.symbol.lower()}@kline_{self.interval}"
            if not testnet
            else f"wss://stream.binancefuture.com/stream?streams={self.symbol.lower()}@kline_{self.interval}"
        )
        self.history: List[Candle] = []

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
