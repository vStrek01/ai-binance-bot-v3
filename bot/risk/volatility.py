"""Volatility snapshots used by the sizing logic."""
from __future__ import annotations

import math
import pandas as pd

from bot.core.config import BotConfig
from bot.signals import indicators


def snapshot(frame: pd.DataFrame, cfg: BotConfig) -> dict[str, float]:
    atr_series = indicators.atr(frame, cfg.sizing.atr_period)
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty and not math.isnan(atr_series.iloc[-1]) else 0.0
    std_window = min(len(frame), cfg.sizing.std_window)
    std_value = float(frame["close"].astype(float).tail(std_window).std()) if std_window else 0.0
    return {"atr": atr_value, "stddev": std_value}


__all__ = ["snapshot"]
