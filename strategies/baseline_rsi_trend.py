from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence

import numpy as np

from core.models import Candle


@dataclass(slots=True)
class BaselineDecision:
    action: str
    size_usd: float
    sl_pct: float
    tp_pct: float


@dataclass(slots=True)
class BaselineConfig:
    ma_length: int = 50
    rsi_length: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    size_usd: float = 1_000.0
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02


class BaselineRSITrend:
    """Deterministic EMA/RSI baseline used for safety comparisons."""

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()

    def evaluate(self, candles: Sequence[Candle]) -> BaselineDecision:
        if len(candles) < max(self.config.ma_length, self.config.rsi_length + 1):
            return self._flat()

        closes = [c.close for c in candles]
        price = closes[-1]
        ma_value = mean(closes[-self.config.ma_length :])
        rsi_value = self._compute_rsi(closes, self.config.rsi_length)
        if rsi_value is None:
            return self._flat()

        if price > ma_value and rsi_value < self.config.rsi_oversold:
            return BaselineDecision("LONG", self.config.size_usd, self.config.stop_loss_pct, self.config.take_profit_pct)

        if price < ma_value and rsi_value > self.config.rsi_overbought:
            return BaselineDecision("SHORT", self.config.size_usd, self.config.stop_loss_pct, self.config.take_profit_pct)

        return self._flat()

    def _flat(self) -> BaselineDecision:
        return BaselineDecision("FLAT", 0.0, self.config.stop_loss_pct, self.config.take_profit_pct)

    @staticmethod
    def _compute_rsi(closes: Sequence[float], period: int) -> float | None:
        if len(closes) <= period:
            return None
        deltas = np.diff(closes)
        gains = np.clip(deltas, a_min=0, a_max=None)
        losses = np.clip(-deltas, a_min=0, a_max=None)
        avg_gain = BaselineRSITrend._wilder_average(gains[-period:], period)
        avg_loss = BaselineRSITrend._wilder_average(losses[-period:], period)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _wilder_average(values: Iterable[float], period: int) -> float:
        arr = np.array(list(values), dtype=float)
        if len(arr) < period:
            return float(arr.mean()) if len(arr) else 0.0
        return float(arr.mean())
