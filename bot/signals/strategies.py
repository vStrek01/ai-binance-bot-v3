"""Strategy definitions shared across backtest, execution, and RL."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from bot.core.config import BotConfig
from bot.signals import indicators


@dataclass(slots=True)
class StrategyParameters:
    fast_ema: int
    slow_ema: int
    rsi_length: int
    rsi_overbought: float
    rsi_oversold: float
    atr_period: int
    atr_stop: float
    atr_target: float
    cooldown_bars: int
    hold_bars: int


@dataclass(slots=True)
class StrategySignal:
    index: int
    direction: int
    entry_price: float
    stop_loss: float
    take_profit: float
    indicators: Dict[str, float] = field(default_factory=dict)


class EmaRsiAtrStrategy:
    def __init__(self, params: StrategyParameters) -> None:
        self.params = params

    def _annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy().reset_index(drop=True)
        enriched["ema_fast"] = indicators.ema(enriched["close"], self.params.fast_ema)
        enriched["ema_slow"] = indicators.ema(enriched["close"], self.params.slow_ema)
        enriched["rsi"] = indicators.rsi(enriched["close"], self.params.rsi_length)
        enriched["atr"] = indicators.atr(enriched, self.params.atr_period)
        return enriched

    def generate_signals(self, frame: pd.DataFrame) -> List[StrategySignal]:
        annotated = self._annotate(frame)
        signals: List[StrategySignal] = []
        cooldown = 0
        for idx, row in enumerate(annotated.itertuples(), start=0):
            if cooldown > 0:
                cooldown -= 1
                continue
            if pd.isna(row.atr):
                continue
            price = float(row.close)
            ema_fast = float(row.ema_fast)
            ema_slow = float(row.ema_slow)
            rsi_value = float(row.rsi)
            atr_value = float(row.atr)
            go_long = ema_fast > ema_slow and rsi_value <= self.params.rsi_oversold
            go_short = ema_fast < ema_slow and rsi_value >= self.params.rsi_overbought
            if go_long:
                stop = price - atr_value * self.params.atr_stop
                target = price + atr_value * self.params.atr_target
                signals.append(
                    StrategySignal(
                        idx,
                        1,
                        price,
                        stop,
                        target,
                        indicators={
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "rsi": rsi_value,
                            "atr": atr_value,
                        },
                    )
                )
                cooldown = self.params.cooldown_bars
            elif go_short:
                stop = price + atr_value * self.params.atr_stop
                target = price - atr_value * self.params.atr_target
                signals.append(
                    StrategySignal(
                        idx,
                        -1,
                        price,
                        stop,
                        target,
                        indicators={
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "rsi": rsi_value,
                            "atr": atr_value,
                        },
                    )
                )
                cooldown = self.params.cooldown_bars
        return signals


def build_parameters(cfg: BotConfig, overrides: Optional[Dict[str, float | int]] = None) -> StrategyParameters:
    settings = cfg.strategy.default_parameters.copy()
    if overrides:
        settings.update(overrides)
    return StrategyParameters(
        fast_ema=int(settings["fast_ema"]),
        slow_ema=int(settings["slow_ema"]),
        rsi_length=int(settings.get("rsi_length", 14)),
        rsi_overbought=float(settings["rsi_overbought"]),
        rsi_oversold=float(settings["rsi_oversold"]),
        atr_period=int(settings.get("atr_period", 14)),
        atr_stop=float(settings["atr_stop"]),
        atr_target=float(settings["atr_target"]),
        cooldown_bars=int(settings.get("cooldown_bars", 3)),
        hold_bars=int(settings.get("hold_bars", 120)),
    )


__all__ = ["StrategyParameters", "StrategySignal", "EmaRsiAtrStrategy", "build_parameters"]
