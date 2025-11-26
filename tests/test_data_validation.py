from typing import Any, Dict

import pandas as pd
import pytest

from bot.data.feeds import CandleValidationError, validate_candles


def _build_frame(rows: int = 3, freq: str = "1min") -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=rows, freq=freq, tz="UTC")
    close_offset = pd.Timedelta(freq)
    data: Dict[str, Any] = {
        "open_time": times,
        "open": [100 + i for i in range(rows)],
        "high": [101 + i for i in range(rows)],
        "low": [99 + i for i in range(rows)],
        "close": [100.5 + i for i in range(rows)],
        "volume": [10.0] * rows,
        "close_time": times + close_offset,
        "quote_volume": [5.0] * rows,
        "trades": [1] * rows,
        "taker_base": [2.0] * rows,
        "taker_quote": [3.0] * rows,
        "ignore": [0] * rows,
        "symbol": ["BTCUSDT"] * rows,
        "interval": ["1m"] * rows,
    }
    return pd.DataFrame(data)


def test_validate_candles_accepts_clean_frame() -> None:
    frame = _build_frame()
    validate_candles(frame, "BTCUSDT", "1m")


def test_validate_candles_detects_duplicate_timestamps() -> None:
    frame = _build_frame()
    frame.loc[1, "open_time"] = frame.loc[0, "open_time"]
    with pytest.raises(CandleValidationError):
        validate_candles(frame, "BTCUSDT", "1m")


def test_validate_candles_detects_gaps() -> None:
    frame = _build_frame()
    frame.loc[2, "open_time"] = frame.loc[1, "open_time"] + pd.Timedelta(minutes=5)
    with pytest.raises(CandleValidationError):
        validate_candles(frame, "BTCUSDT", "1m")


def test_validate_candles_detects_negative_prices() -> None:
    frame = _build_frame()
    frame.loc[0, "open"] = -1
    with pytest.raises(CandleValidationError):
        validate_candles(frame, "BTCUSDT", "1m")


def test_validate_candles_detects_nans() -> None:
    frame = _build_frame()
    frame.loc[0, "close"] = None
    with pytest.raises(CandleValidationError):
        validate_candles(frame, "BTCUSDT", "1m")
