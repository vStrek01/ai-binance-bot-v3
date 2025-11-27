"""Shared trade simulation helpers for signal evaluation tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, cast

import pandas as pd

from bot.signals.strategies import StrategySignal


@dataclass
class SimulatedTrade:
    index: int
    open_time: pd.Timestamp
    direction: int
    entry: float
    stop: float
    target: float
    close: float
    close_time: pd.Timestamp
    outcome: str
    r_multiple: float


def _normalize_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    timestamp = cast(pd.Timestamp, timestamp)
    if pd.isna(timestamp):
        timestamp = pd.to_datetime(pd.Timestamp.utcnow(), utc=True)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _row_timestamp(df: pd.DataFrame, idx: int) -> pd.Timestamp:
    if "close_time" in df.columns:
        return _normalize_timestamp(df.iloc[idx]["close_time"])
    if "open_time" in df.columns:
        return _normalize_timestamp(df.iloc[idx]["open_time"])
    return _normalize_timestamp(df.index[idx])


def simulate_trades(df: pd.DataFrame, signals: List[StrategySignal]) -> List[SimulatedTrade]:
    """Replay ATR stop/target outcomes for each signal using raw candle data."""

    if df.empty or not signals:
        return []

    trades: List[SimulatedTrade] = []
    total_rows = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    for signal in signals:
        idx = int(signal.index)
        if idx >= total_rows:
            idx = total_rows - 1
        if idx < 0:
            continue

        direction = signal.direction or 0
        entry = float(signal.entry_price)
        stop = float(signal.stop_loss)
        target = float(signal.take_profit)
        open_time = _row_timestamp(df, idx)
        outcome = "FLAT"
        close_price = float(closes[-1])
        close_time = _row_timestamp(df, total_rows - 1)

        for j in range(idx + 1, total_rows):
            bar_high = float(highs[j])
            bar_low = float(lows[j])
            timestamp = _row_timestamp(df, j)

            if direction == 1:
                stop_hit = bar_low <= stop
                target_hit = bar_high >= target
                if stop_hit:
                    outcome = "SL"
                    close_price = stop
                    close_time = timestamp
                    break
                if target_hit:
                    outcome = "TP"
                    close_price = target
                    close_time = timestamp
                    break
            else:
                stop_hit = bar_high >= stop
                target_hit = bar_low <= target
                if stop_hit:
                    outcome = "SL"
                    close_price = stop
                    close_time = timestamp
                    break
                if target_hit:
                    outcome = "TP"
                    close_price = target
                    close_time = timestamp
                    break
        else:
            outcome = "FLAT"

        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0 or direction == 0:
            r_multiple = 0.0
        else:
            r_multiple = (close_price - entry) / risk_per_unit * direction

        trades.append(
            SimulatedTrade(
                index=idx,
                open_time=open_time,
                direction=direction,
                entry=entry,
                stop=stop,
                target=target,
                close=close_price,
                close_time=close_time,
                outcome=outcome,
                r_multiple=r_multiple,
            )
        )

    return trades
