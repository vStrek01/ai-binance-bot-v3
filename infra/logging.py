import logging
import os
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting logic
        log_record: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_record.update(record.extra)
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return str(log_record)


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("ai_binance_bot")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = JsonFormatter()
    handler.setFormatter(formatter)

    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _configure_logger()
