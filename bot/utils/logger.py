"""Logging helper for the bot."""
from __future__ import annotations

import logging

from bot import config

_logger_initialized = False


def get_logger(name: str) -> logging.Logger:
    global _logger_initialized
    if not _logger_initialized:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.LOG_DIR / "bot.log"
        handlers = [logging.StreamHandler()]
        file_handler_error: str | None = None
        try:
            handlers.insert(0, logging.FileHandler(log_path, encoding="utf-8"))
        except OSError as exc:
            file_handler_error = f"File logging disabled for {log_path}: {exc}"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            handlers=handlers,
        )
        _logger_initialized = True
        if file_handler_error:
            logging.getLogger(__name__).warning(file_handler_error)
    return logging.getLogger(name)

__all__ = ["get_logger"]
