"""Shared helpers for parsing structured log_event outputs."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, Iterator, Optional

DEFAULT_LOG_PATH = Path(os.getenv("LOG_FILE", "logs/bot.log"))


@dataclass(slots=True)
class LogEvent:
    """Normalized representation of a structured log entry."""

    recorded_ts: str
    event: str
    payload: Dict[str, object]
    raw: str

    @property
    def symbol(self) -> Optional[str]:
        symbol = self.payload.get("symbol")
        return str(symbol) if symbol is not None else None


def parse_log_line(line: str) -> Optional[LogEvent]:
    """Return a LogEvent when the line contains JSON with an 'event' field."""

    brace = line.find("{")
    if brace == -1:
        return None
    prefix = line[:brace].strip()
    body = line[brace:]
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    event = payload.get("event")
    if not isinstance(event, str) or not event:
        return None
    recorded_ts = str(payload.get("timestamp") or prefix)
    return LogEvent(recorded_ts=recorded_ts, event=event, payload=payload, raw=line.rstrip("\n"))


def iter_log_events(path: Path) -> Iterator[LogEvent]:
    """Yield parsed LogEvent objects by reading the file once from start to finish."""

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                event = parse_log_line(line)
                if event:
                    yield event
    except FileNotFoundError:
        return


def follow_log_events(
    path: Path,
    *,
    poll_interval: float = 1.0,
    seek_end: bool = True,
) -> Generator[LogEvent, None, None]:
    """Yield events as they appear, similar to `tail -f` semantics."""

    poll_interval = max(0.25, poll_interval)
    while True:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                if seek_end:
                    handle.seek(0, os.SEEK_END)
                while True:
                    line = handle.readline()
                    if not line:
                        time.sleep(poll_interval)
                        if not path.exists():
                            break
                        continue
                    event = parse_log_line(line)
                    if event:
                        yield event
        except FileNotFoundError:
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            raise
        except Exception:
            # Avoid tight failure loops; loggers aren't available in standalone tools.
            time.sleep(poll_interval)


ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)


def coerce_timestamp(raw: str) -> Optional[datetime]:
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if "T" not in raw and " " in raw:
        raw = raw.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass
    for fmt in ISO_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


__all__ = [
    "DEFAULT_LOG_PATH",
    "LogEvent",
    "coerce_timestamp",
    "parse_log_line",
    "iter_log_events",
    "follow_log_events",
]
