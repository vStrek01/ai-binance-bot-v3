from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from core.llm_adapter import LLMAdapter
from core.models import Candle, LLMSignal, MarketState, Signal, Side
from infra.logging import logger


@dataclass
class IndicatorConfig:
    fast_ma: int = 9
    slow_ma: int = 21
    min_confidence: float = 0.55


class Strategy:
    def __init__(self, indicator_config: IndicatorConfig, llm_adapter: Optional[LLMAdapter] = None):
        self.config = indicator_config
        self.llm_adapter = llm_adapter

    def _compute_ma_signal(self, candles: List[Candle]) -> Signal:
        closes = [c.close for c in candles]
        if len(closes) < self.config.slow_ma:
            return Signal(action=Side.FLAT, confidence=0.0, reason="insufficient data")

        close_series = pd.Series(closes)
        fast = close_series.rolling(self.config.fast_ma).mean().iloc[-1]
        slow = close_series.rolling(self.config.slow_ma).mean().iloc[-1]

        if np.isnan(fast) or np.isnan(slow):
            return Signal(action=Side.FLAT, confidence=0.0, reason="insufficient data")

        if fast > slow:
            return Signal(action=Side.LONG, confidence=0.6, reason="fast MA above slow")
        if fast < slow:
            return Signal(action=Side.SHORT, confidence=0.6, reason="fast MA below slow")
        return Signal(action=Side.FLAT, confidence=0.0, reason="flat MAs")

    def _fuse_signals(self, indicator_signal: Signal, llm_signal: Optional[LLMSignal]) -> Signal:
        if llm_signal is None:
            return indicator_signal

        if indicator_signal.action == Side.FLAT:
            return llm_signal

        if llm_signal.action == Side.FLAT:
            return indicator_signal

        if indicator_signal.action == llm_signal.action:
            avg_conf = (indicator_signal.confidence + llm_signal.confidence) / 2
            return Signal(action=indicator_signal.action, confidence=avg_conf, reason="agreement")

        # disagreement -> reduce conviction
        return Signal(action=Side.FLAT, confidence=0.0, reason="disagreement")

    def evaluate(self, market_state: MarketState) -> Signal:
        indicator_signal = self._compute_ma_signal(market_state.candles)
        llm_signal = None
        if self.llm_adapter:
            context = {
                "symbol": market_state.symbol,
                "last_close": market_state.candles[-1].close,
                "equity": market_state.equity,
            }
            llm_signal = self.llm_adapter.infer(context)

        fused = self._fuse_signals(indicator_signal, llm_signal)
        if fused.confidence < self.config.min_confidence:
            logger.info("Signal confidence too low", extra={"signal": fused.model_dump()})
            return Signal(action=Side.FLAT, confidence=0.0, reason="low confidence")
        return fused
