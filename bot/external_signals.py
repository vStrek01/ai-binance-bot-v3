"""Optional sentiment, news, and on-chain signal helpers."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from bot.core.config import BotConfig
from bot.data import fetch_recent_candles
from bot.signals import indicators
from bot.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ExternalSignalSnapshot:
    sentiment: float = 0.0
    news: float = 0.0
    onchain: float = 0.0
    combined: float = 0.0
    regime: str = "neutral"
    warnings: tuple[str, ...] = ()


class ExternalSignalProvider:
    """Lightweight signal aggregator with pluggable data sources."""

    def __init__(self, cfg: BotConfig) -> None:
        self._config = cfg
        self.sentiment_file = os.getenv("BOT_SENTIMENT_FILE")
        self.news_file = os.getenv("BOT_NEWS_FILE")
        self.onchain_file = os.getenv("BOT_ONCHAIN_FILE")

    def snapshot(self, symbol: str) -> ExternalSignalSnapshot:
        cfg = self._config.external_signals
        if not cfg.enabled:
            return ExternalSignalSnapshot()
        warnings: list[str] = []
        sentiment = self._load_scalar(self.sentiment_file, symbol, warnings)
        news = self._load_scalar(self.news_file, symbol, warnings)
        onchain = self._load_scalar(self.onchain_file, symbol, warnings)
        if math.isclose(sentiment, 0.0) and math.isclose(news, 0.0) and math.isclose(onchain, 0.0):
            # Fallback to price-derived proxy when no external data is provided.
            sentiment, news, onchain = self._price_proxy_scores(symbol, warnings)
        combined = sentiment * cfg.sentiment_weight + news * cfg.news_weight + onchain * cfg.onchain_weight
        regime = self._detect_regime(symbol, warnings)
        snapshot = ExternalSignalSnapshot(
            sentiment=sentiment,
            news=news,
            onchain=onchain,
            combined=combined,
            regime=regime,
            warnings=tuple(warnings),
        )
        return snapshot

    def _load_scalar(
        self,
        source: Optional[str],
        symbol: str,
        warnings: list[str],
    ) -> float:
        if not source:
            return 0.0
        path = Path(source)
        if not path.exists():
            warnings.append(f"missing_source:{path.name}")
            return 0.0
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            warnings.append(f"invalid_source:{path.name}")
            return 0.0
        value = payload.get(symbol) if isinstance(payload, dict) else None
        if value is None:
            return 0.0
        try:
            score = float(value)
        except (TypeError, ValueError):
            warnings.append(f"bad_value:{path.name}")
            return 0.0
        return max(-1.0, min(1.0, score))

    def _price_proxy_scores(self, symbol: str, warnings: list[str]) -> tuple[float, float, float]:
        try:
            candles = fetch_recent_candles(self._config, symbol, "5m", limit=120)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"price_proxy_failed:{exc}")
            return 0.0, 0.0, 0.0
        closes = candles["close"].astype(float)
        returns = pd.Series(closes).pct_change().dropna()
        if returns.empty:
            return 0.0, 0.0, 0.0
        momentum = returns.tail(12).mean()
        volatility = returns.tail(60).std()
        skew = returns.skew() if hasattr(returns, "skew") else 0.0
        sentiment = max(-1.0, min(1.0, momentum * 50))
        news = max(-1.0, min(1.0, skew * 10))
        onchain = max(-1.0, min(1.0, (0.02 - volatility) * 20))
        return sentiment, news, onchain

    def _detect_regime(self, symbol: str, warnings: list[str]) -> str:
        try:
            candles = fetch_recent_candles(self._config, symbol, "15m", limit=200)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"regime_failed:{exc}")
            return "neutral"
        ema_fast = indicators.ema(candles["close"], 34)
        ema_slow = indicators.ema(candles["close"], 89)
        atr = indicators.atr(candles, 14)
        if ema_fast.empty or ema_slow.empty or atr.empty:
            return "neutral"
        trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        vol = atr.iloc[-1] / candles["close"].iloc[-1]
        if vol > 0.02:
            return "volatile_up" if trend_up else "volatile_down"
        if trend_up:
            return "trend_up"
        if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            return "trend_down"
        return "neutral"


__all__ = ["ExternalSignalProvider", "ExternalSignalSnapshot"]
