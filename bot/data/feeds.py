"""Data acquisition utilities: download, caching, validation."""
from __future__ import annotations

from typing import List, Optional, Sequence, cast

import pandas as pd

from bot.core.config import BotConfig, ensure_directories
from bot.execution.client_factory import build_data_client
from bot.execution.exchange_client import ExchangeRequestError
from bot.utils.logger import get_logger

logger = get_logger(__name__)

KLINE_COLUMNS: Sequence[str] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
)
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote")
REQUIRED_COLUMNS = ("open_time", "close_time", "open", "high", "low", "close", "volume")


class CandleValidationError(ValueError):
    """Raised when candle data fails integrity checks."""


def _interval_to_timedelta(interval: str) -> Optional[pd.Timedelta]:
    if not interval:
        return None
    unit = interval[-1].lower()
    try:
        value = int(interval[:-1])
    except ValueError:
        return None
    lookup = {"m": "min", "h": "h", "d": "d", "w": "w"}
    suffix = lookup.get(unit)
    if suffix is None or value <= 0:
        return None
    return pd.to_timedelta(f"{value}{suffix}")


def _klines_to_frame(rows: Sequence[Sequence[str]], symbol: str, interval: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    frame["symbol"] = symbol
    frame["interval"] = interval
    frame[list(NUMERIC_COLUMNS)] = frame[list(NUMERIC_COLUMNS)].astype(float)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    return frame


def download_klines(
    cfg: BotConfig,
    symbol: str,
    interval: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1500,
    save: bool = True,
) -> pd.DataFrame:
    client = build_data_client(cfg)
    try:
        rows = cast(
            List[List[str]],
            client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
                limit=limit,
            ),
        )
    except ExchangeRequestError as exc:
        logger.error("Kline download failed for %s %s: %s", symbol, interval, exc)
        raise

    frame = _klines_to_frame(rows, symbol, interval)
    validate_candles(frame, symbol, interval)
    if save:
        ensure_directories(cfg.paths)
        path = cfg.paths.data_dir / f"{symbol}_{interval}.csv"
        frame.to_csv(path, index=False)
        logger.info("Saved %s rows to %s", len(frame), path)
    return frame


def load_local_candles(cfg: BotConfig, symbol: str, interval: str) -> pd.DataFrame:
    path = cfg.paths.data_dir / f"{symbol}_{interval}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cached candles at {path}")
    frame = pd.read_csv(path, parse_dates=["open_time", "close_time"])  # type: ignore[arg-type]
    validate_candles(frame, symbol, interval)
    return frame


def fetch_recent_candles(cfg: BotConfig, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    client = build_data_client(cfg)
    rows = cast(
        List[List[str]],
        client.get_klines(symbol=symbol, interval=interval, limit=limit),
    )
    frame = _klines_to_frame(rows, symbol, interval)
    validate_candles(frame, symbol, interval)
    return frame


def ensure_local_candles(
    cfg: BotConfig,
    symbol: str,
    interval: str,
    min_rows: int = 1_000,
    download_limit: Optional[int] = None,
) -> pd.DataFrame:
    path = cfg.paths.data_dir / f"{symbol}_{interval}.csv"
    if path.exists():
        frame = load_local_candles(cfg, symbol, interval)
        if len(frame) >= min_rows:
            return frame
        logger.info(
            "Cached candles for %s %s below threshold (%s < %s); refreshing",
            symbol,
            interval,
            len(frame),
            min_rows,
        )
    limit = download_limit or max(min_rows, 1500)
    return download_klines(cfg, symbol, interval, limit=limit, save=True)


def validate_candles(frame: pd.DataFrame, symbol: str, interval: str) -> None:
    if frame.empty:
        raise CandleValidationError(f"{symbol} {interval} dataset is empty")
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise CandleValidationError(f"{symbol} {interval} missing columns: {', '.join(missing)}")
    if not frame["open_time"].is_monotonic_increasing:
        raise CandleValidationError(f"{symbol} {interval} timestamps out of order")
    duplicates = frame["open_time"].duplicated(keep=False)
    if duplicates.any():
        idx = duplicates[duplicates].index[:5].tolist()
        raise CandleValidationError(f"{symbol} {interval} duplicate timestamps at rows {idx}")
    expected_delta = _interval_to_timedelta(interval)
    if expected_delta is not None and len(frame) > 1:
        deltas = frame["open_time"].diff().iloc[1:]
        gaps = deltas != expected_delta
        if gaps.any():
            idx = gaps[gaps].index[:5].tolist()
            raise CandleValidationError(
                f"{symbol} {interval} has timestamp gaps at rows {idx} (expected {expected_delta})"
            )
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise CandleValidationError(f"{symbol} {interval} contains NaN values")
    price_cols = ["open", "high", "low", "close"]
    if (frame[price_cols] <= 0).any().any():
        raise CandleValidationError(f"{symbol} {interval} contains non-positive prices")
    open_close = frame[["open", "close"]]
    high_ok = frame["high"] >= open_close.max(axis=1)
    low_ok = frame["low"] <= open_close.min(axis=1)
    if not (high_ok & low_ok).all():
        bad_idx = frame.index[~(high_ok & low_ok)][:5].tolist()
        raise CandleValidationError(f"{symbol} {interval} has inconsistent high/low values at rows {bad_idx}")
    if (frame["volume"] < 0).any():
        raise CandleValidationError(f"{symbol} {interval} contains negative volume values")


__all__ = [
    "download_klines",
    "load_local_candles",
    "fetch_recent_candles",
    "ensure_local_candles",
    "validate_candles",
    "CandleValidationError",
]
