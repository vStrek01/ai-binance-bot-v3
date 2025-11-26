"""Structured logging helper for the bot."""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - import guard for typing
    from bot.core.config import BotConfig


@dataclass(frozen=True)
class RunContext:
    """Runtime context that propagates run correlation identifiers."""

    run_id: str
    run_type: str = "general"


_logger_initialized = False
_current_context: Optional[RunContext] = None


class RunContextFilter(logging.Filter):
    """Inject run identifiers into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - logging API
        record.run_id = getattr(_current_context, "run_id", "-")
        record.run_type = getattr(_current_context, "run_type", "unknown")
        return True


class JSONFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - logging API
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
            "run_type": getattr(record, "run_type", "unknown"),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                continue
            if key in payload:
                continue
            try:
                json.dumps({key: value})
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
        return json.dumps(payload)


def generate_run_id(prefix: str = "run") -> str:
    """Create a short unique run identifier."""

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    suffix = os.urandom(3).hex()
    return f"{prefix}-{ts}-{suffix}"


def set_run_context(ctx: RunContext) -> None:
    """Update the shared run context."""

    global _current_context
    _current_context = ctx


def get_run_context() -> Optional[RunContext]:
    """Return the current run context if set."""

    return _current_context


def _resolve_log_dir(cfg: "BotConfig", run_type: str) -> Path:
    base = cfg.paths.log_dir if cfg else Path(os.getenv("BOT_LOG_DIR", "logs"))
    return (base / run_type).resolve()


def setup_logging(
    cfg: "BotConfig",
    *,
    run_context: RunContext,
    format_override: Optional[str] = None,
    level_override: Optional[str] = None,
) -> None:
    """Configure root logging with structured, per-run outputs."""

    global _logger_initialized
    set_run_context(run_context)
    log_format = (format_override or cfg.runtime.log_format or "text").lower()
    level_name = (level_override or cfg.runtime.log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = _resolve_log_dir(cfg, run_context.run_type)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / f"{run_context.run_id}.log"

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)

    original_factory = logging.getLogRecordFactory()

    def _factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = original_factory(*args, **kwargs)
        record.run_id = getattr(_current_context, "run_id", "-")
        record.run_type = getattr(_current_context, "run_type", "unknown")
        return record

    logging.setLogRecordFactory(_factory)

    formatter: logging.Formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(run_type)s | %(run_id)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    filter_ = RunContextFilter()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - filesystem guard
        logging.getLogger(__name__).warning("File logging disabled for %s: %s", file_path, exc)

    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        handler.addFilter(filter_)
        root.addHandler(handler)

    root.addFilter(filter_)
    _logger_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger; assumes :func:`setup_logging` is called once by the entrypoint."""

    if not _logger_initialized:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    return logging.getLogger(name)


__all__ = ["RunContext", "generate_run_id", "get_logger", "get_run_context", "set_run_context", "setup_logging"]
