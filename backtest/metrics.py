from __future__ import annotations

import math
from typing import List

import numpy as np


def sharpe_ratio(returns: List[float], risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = np.array(returns) - risk_free
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * math.sqrt(252)


def max_drawdown(equity_curve: List[float]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)
    return max_dd


def win_rate(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    wins = len([p for p in pnls if p > 0])
    return wins / len(pnls)


def average_r(pnls: List[float], risks: List[float]) -> float:
    if not pnls or not risks or len(pnls) != len(risks):
        return 0.0
    rs = [p / r if r else 0 for p, r in zip(pnls, risks)]
    return float(np.mean(rs))


def exposure(open_bars: List[int], total_bars: int) -> float:
    if total_bars == 0:
        return 0.0
    return sum(open_bars) / total_bars
