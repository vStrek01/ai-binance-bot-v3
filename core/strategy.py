from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.llm_adapter import LLMAdapter
from core.models import Candle, LLMSignal, MarketState, Signal, Side
from infra.logging import logger


@dataclass
class IndicatorConfig:
    fast_ema: int = 13
    slow_ema: int = 34
    rsi_length: int = 14
    rsi_overbought: float = 60.0
    rsi_oversold: float = 40.0
    atr_period: int = 14
    atr_stop: float = 1.6
    atr_target: float = 2.2
    cooldown_bars: int = 2
    hold_bars: int = 90
    pullback_atr_multiplier: float = 1.0
    pullback_rsi_threshold: float = 50.0
    trend_strength_threshold: float = 0.3
    min_confidence: float = 0.55


class Strategy:
    def __init__(
        self,
        indicator_config: IndicatorConfig,
        llm_adapter: Optional[LLMAdapter] = None,
        optimized_params: Optional[Dict[str, Any]] = None,
    ):
        self._base_config = indicator_config
        self.llm_adapter = llm_adapter
        self.optimized_params = optimized_params or {}
        self.config = self._merge_config(self._base_config, self.optimized_params)

    def _merge_config(self, config: IndicatorConfig, overrides: Dict[str, Any]) -> IndicatorConfig:
        if not overrides:
            return config
        config_dict = asdict(config)
        for key, value in overrides.items():
            if key in config_dict and value is not None:
                config_dict[key] = value
        return IndicatorConfig(**config_dict)

    def _compute_ma_signal(self, candles: List[Candle]) -> Signal:
        closes = [c.close for c in candles]
        if len(closes) < self.config.slow_ema:
            return Signal(action=Side.FLAT, confidence=0.0, reason="insufficient data")

        close_series = pd.Series(closes)
        fast = close_series.rolling(self.config.fast_ema).mean().iloc[-1]
        slow = close_series.rolling(self.config.slow_ema).mean().iloc[-1]

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
