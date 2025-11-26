"""Risk filters such as multi-timeframe confirmations and external gates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from bot.core.config import BotConfig
from bot.external_signals import ExternalSignalProvider
from bot.signals import indicators


@dataclass(slots=True)
class MultiTimeframeFilter:
    cfg: BotConfig
    candle_source: Callable[[str, str, int], Any]

    def evaluate(self, symbol: str, direction: int) -> Tuple[bool, Dict[str, Any]]:
        cfg = self.cfg.multi_timeframe
        if not cfg.enabled or not cfg.confirm_timeframes:
            return True, {}
        confirmations: Dict[str, Any] = {}
        passed = True if cfg.require_alignment else False
        for timeframe in cfg.confirm_timeframes:
            try:
                candles = self.candle_source(symbol, timeframe, min(self.cfg.runtime.lookback_limit, 500))
            except Exception as exc:  # noqa: BLE001
                confirmations[timeframe] = {"status": f"error:{exc}"}
                if cfg.require_alignment:
                    passed = False
                continue
            ema_fast = indicators.ema(candles["close"], cfg.ema_fast)
            ema_slow = indicators.ema(candles["close"], cfg.ema_slow)
            rsi_series = indicators.rsi(candles["close"], 14)
            if ema_fast.empty or ema_slow.empty or rsi_series.empty:
                confirmations[timeframe] = {"status": "insufficient"}
                if cfg.require_alignment:
                    passed = False
                continue
            trend_up = ema_fast.iloc[-1] > ema_slow.iloc[-1]
            rsi_value = float(rsi_series.iloc[-1])
            if direction == 1:
                tf_passed = trend_up and rsi_value <= cfg.rsi_upper
            else:
                tf_passed = (not trend_up) and rsi_value >= cfg.rsi_lower
            confirmations[timeframe] = {
                "trend_up": trend_up,
                "rsi": rsi_value,
                "status": "pass" if tf_passed else "fail",
            }
            if cfg.require_alignment:
                passed = passed and tf_passed
            else:
                passed = passed or tf_passed
        if not confirmations:
            return True, {}
        return passed, confirmations


class ExternalSignalGate:
    def __init__(self, cfg: BotConfig, provider: ExternalSignalProvider | None = None) -> None:
        self._config = cfg
        self.provider = provider or ExternalSignalProvider()
        self._latest_scores: Dict[str, float] = {}

    @property
    def latest_scores(self) -> Dict[str, float]:
        return dict(self._latest_scores)

    def evaluate(self, symbol: str, direction: int) -> Tuple[bool, Dict[str, Any]]:
        cfg = self._config.external_signals
        if not cfg.enabled:
            return True, {}
        snapshot = self.provider.snapshot(symbol)
        self._latest_scores[symbol] = snapshot.combined
        if snapshot.combined <= cfg.suppression_threshold:
            return False, {"score": snapshot.combined, "regime": snapshot.regime, "flags": snapshot.warnings}
        if direction == -1 and snapshot.sentiment > cfg.boost_threshold:
            return False, {"score": snapshot.combined, "regime": snapshot.regime, "flags": ("sentiment_block",)}
        return True, {
            "score": snapshot.combined,
            "sentiment": snapshot.sentiment,
            "news": snapshot.news,
            "onchain": snapshot.onchain,
            "regime": snapshot.regime,
            "flags": snapshot.warnings,
        }


__all__ = ["MultiTimeframeFilter", "ExternalSignalGate"]
