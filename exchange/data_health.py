from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

from infra.logging import log_event


def _default_clock() -> float:
    return time.time()


def _to_timestamp(value: float | datetime | None) -> float:
    if value is None:
        return _default_clock()
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.timestamp())


def interval_to_seconds(interval: str) -> float:
    if not interval:
        return 60.0
    unit = interval[-1].lower()
    try:
        value = float(interval[:-1])
    except ValueError:
        return 60.0
    multiplier = {"s": 1.0, "m": 60.0, "h": 3_600.0, "d": 86_400.0, "w": 604_800.0}.get(unit)
    if multiplier is None:
        return 60.0
    seconds = value * multiplier
    return max(seconds, 1.0)


@dataclass(frozen=True)
class DataHealthStatus:
    healthy: bool
    seconds_since_update: Optional[float]
    threshold_seconds: float
    last_update: Optional[float]


class DataHealthMonitor:
    def __init__(
        self,
        *,
        stale_multiplier: float = 2.5,
        interval_overrides: Optional[Dict[str, float]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._stale_multiplier = max(stale_multiplier, 1.0)
        self._interval_overrides = interval_overrides or {}
        self._clock: Callable[[], float] = clock or _default_clock
        self._last_updates: Dict[Tuple[str, str], float] = {}
        self._last_status: Dict[Tuple[str, str], bool] = {}
        self._lock = threading.Lock()

    def mark_update(self, symbol: str, interval: str, *, timestamp: float | datetime | None = None) -> None:
        key = self._key(symbol, interval)
        value = _to_timestamp(timestamp)
        with self._lock:
            self._last_updates[key] = value
            self._last_status[key] = True

    def is_data_stale(self, symbol: str, interval: str, *, now: Optional[float] = None) -> DataHealthStatus:
        key = self._key(symbol, interval)
        current = self._now(now)
        with self._lock:
            last_update = self._last_updates.get(key)
        if last_update is None:
            return DataHealthStatus(False, None, self._threshold(interval), None)
        threshold = self._threshold(interval)
        delta = max(current - last_update, 0.0)
        healthy = delta <= threshold
        return DataHealthStatus(healthy, delta, threshold, last_update)

    def is_healthy(self, symbol: str, interval: str, *, now: Optional[float] = None) -> bool:
        status = self.is_data_stale(symbol, interval, now=now)
        key = self._key(symbol, interval)
        with self._lock:
            previous = self._last_status.get(key)
            self._last_status[key] = status.healthy
        if not status.healthy and previous is not False:
            log_event(
                "DATA_STALE",
                symbol=symbol,
                interval=interval,
                seconds_since_update=status.seconds_since_update,
                threshold_seconds=status.threshold_seconds,
            )
        return status.healthy

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            items = dict(self._last_updates)
        return {f"{symbol}:{interval}": value for (symbol, interval), value in items.items()}

    def _threshold(self, interval: str) -> float:
        return self._interval_overrides.get(interval, interval_to_seconds(interval) * self._stale_multiplier)

    def _now(self, override: Optional[float]) -> float:
        return float(override) if override is not None else self._clock()

    @staticmethod
    def _key(symbol: str, interval: str) -> Tuple[str, str]:
        return (symbol.upper(), interval)



_global_monitor: Optional[DataHealthMonitor] = None


def get_data_health_monitor() -> DataHealthMonitor:
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = DataHealthMonitor()
    return _global_monitor


__all__ = [
    "DataHealthMonitor",
    "DataHealthStatus",
    "get_data_health_monitor",
    "interval_to_seconds",
]
