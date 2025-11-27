from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

_LOGGER_NAME = os.getenv("LOG_NAME", "ai_binance_bot")
logger = logging.getLogger(_LOGGER_NAME)

# Central catalogue of structured observability events. Used to stamp
# a stable category field onto every JSON log for downstream pipelines.
EVENT_CATEGORY_MAP: Dict[str, str] = {
    "health_check_passed": "health",
    "health_check_failed": "health",
    "state_saved": "state",
    "state_loaded": "state",
    "risk_state_updated": "risk",
    "runtime_mode_resolved": "runtime",
    "runtime_mode_blocked": "runtime",
    "runtime_bootstrap_complete": "runtime",
    "exchange_filters_applied": "exchange",
    "EQUITY_SNAPSHOT": "telemetry",
    "POSITION_OPENED": "telemetry",
    "POSITION_CLOSED": "telemetry",
    "ORDER_PLACED": "execution",
    "ORDER_FILLED": "execution",
    "ORDER_FAILED": "execution",
    "KILL_SWITCH_TRIGGERED": "risk",
    "risk_kill_switch_triggered": "risk",
    "order_rejected_by_risk": "risk",
}

_context: Dict[str, Any] = {}
_configured = False
_state_file: Optional[Path] = None
_state_lock = Lock()
_recent_events = deque(maxlen=int(os.getenv("LOG_EVENT_BUFFER", "50")))
_positions: Dict[str, Dict[str, Any]] = {}
_equity_snapshot: Optional[Dict[str, Any]] = None
_backtest_summary: Optional[Dict[str, Any]] = None


def setup_logging(level: int = logging.INFO, *, log_to_file: bool | str = False, state_file: Optional[str] = None) -> None:
    """Configure the shared logger and optional dashboard state sink."""

    global _configured, _state_file

    if not _configured:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.propagate = False
        logger.addHandler(handler)

        if log_to_file:
            file_path = _resolve_log_file(log_to_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        _configured = True

    desired_state_file = state_file or os.getenv("DASHBOARD_STATE_FILE")
    if desired_state_file:
        _state_file = Path(desired_state_file)


def bind_log_context(**fields: Any) -> None:
    """Attach process-wide contextual fields (e.g. run_mode, run_id)."""

    _context.update({key: value for key, value in fields.items() if value is not None})


def log_event(event: str, **fields: Any) -> None:
    """Emit a JSON log entry tagged with the supplied event name."""

    payload: Dict[str, Any] = {**_context, **fields}
    payload["event"] = event
    category = EVENT_CATEGORY_MAP.get(event)
    if category:
        payload.setdefault("category", category)
    payload.setdefault("timestamp", datetime.utcnow().isoformat())
    logger.info(json.dumps(payload, default=str))
    _update_state(payload)


def _resolve_log_file(log_to_file: bool | str) -> Path:
    if isinstance(log_to_file, str):
        return Path(log_to_file)
    env_path = os.getenv("LOG_FILE")
    return Path(env_path or "logs/bot.log")


def _update_state(payload: Dict[str, Any]) -> None:
    if _state_file is None:
        return

    event = payload.get("event")
    global _backtest_summary
    with _state_lock:
        if event == "EQUITY_SNAPSHOT":
            _record_equity(payload)
        if event in {"POSITION_OPENED", "POSITION_CLOSED"}:
            _record_position(payload)
        if event == "BACKTEST_SUMMARY":
            _backtest_summary = payload

        if event in {
            "ORDER_PLACED",
            "ORDER_FILLED",
            "ORDER_CANCELLED",
            "POSITION_OPENED",
            "POSITION_CLOSED",
            "EQUITY_SNAPSHOT",
            "KILL_SWITCH_TRIGGERED",
            "LLM_INVALID_OUTPUT",
            "BACKTEST_SUMMARY",
        }:
            _recent_events.appendleft({k: v for k, v in payload.items() if k != "position"})

        data = {
            "equity": _equity_snapshot,
            "positions": list(_positions.values()),
            "recent_events": list(_recent_events),
            "backtest_summary": _backtest_summary,
            "updated_at": payload.get("timestamp"),
        }

        _state_file.parent.mkdir(parents=True, exist_ok=True)
        _state_file.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")


def _record_equity(payload: Dict[str, Any]) -> None:
    global _equity_snapshot
    _equity_snapshot = payload


def _record_position(payload: Dict[str, Any]) -> None:
    position = payload.get("position")
    symbol = payload.get("symbol") or (position or {}).get("symbol")
    if not symbol:
        return

    if payload["event"] == "POSITION_CLOSED":
        _positions.pop(symbol, None)
        return

    if position is not None:
        _positions[symbol] = position


def get_recent_events(*, limit: int = 10) -> List[Dict[str, Any]]:
    with _state_lock:
        return list(list(_recent_events)[:limit])


def get_open_positions() -> List[Dict[str, Any]]:
    with _state_lock:
        return list(_positions.values())


def get_equity_snapshot() -> Optional[Dict[str, Any]]:
    with _state_lock:
        return dict(_equity_snapshot) if _equity_snapshot else None


default_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
default_level = getattr(logging, default_level_name, logging.INFO)
setup_logging(default_level)

__all__ = [
    "logger",
    "setup_logging",
    "log_event",
    "bind_log_context",
    "get_recent_events",
    "get_open_positions",
    "get_equity_snapshot",
]
