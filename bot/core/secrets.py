"""Helpers for loading API credentials from environment or optional .env files."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = BASE_DIR / ".env"
_BINANCE_KEY_VARS = ("BINANCE_API_KEY", "BINANCE_KEY")
_BINANCE_SECRET_VARS = ("BINANCE_API_SECRET", "BINANCE_SECRET")
_TESTNET_KEY_VARS = ("BINANCE_TESTNET_API_KEY", "BINANCE_SANDBOX_API_KEY")
_TESTNET_SECRET_VARS = ("BINANCE_TESTNET_API_SECRET", "BINANCE_SANDBOX_API_SECRET")
_dotenv_loaded = False


def _maybe_load_dotenv() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    candidate = os.getenv("BOT_ENV_FILE")
    paths: list[Path] = []
    if candidate:
        paths.append(Path(candidate))
    paths.append(DEFAULT_ENV_FILE)
    for path in paths:
        if not path.exists():
            continue
        try:
            from dotenv import load_dotenv  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - optional dependency
            logger.debug("python-dotenv not installed; skipping .env loading from %s", path)
            return
        load_dotenv(dotenv_path=path, override=False)
        logger.info("Loaded environment variables from %s", path)
        return


def _resolve_env_var(candidates: Iterable[str]) -> str:
    _maybe_load_dotenv()
    for name in candidates:
        value = os.getenv(name)
        if value:
            return value.strip()
    return ""


class SecretNotConfiguredError(RuntimeError):
    """Raised when a required secret is missing."""


@lru_cache(maxsize=None)
def get_binance_api_key(*, prefer_testnet: bool = False, required: bool = False) -> str:
    """Return the configured Binance API key, optionally preferring the testnet variants."""
    candidates: list[str] = []
    if prefer_testnet:
        candidates.extend(_TESTNET_KEY_VARS)
    candidates.extend(_BINANCE_KEY_VARS)
    value = _resolve_env_var(candidates)
    if not value and required:
        raise SecretNotConfiguredError(
            "Missing Binance API key. Set BINANCE_API_KEY or BINANCE_TESTNET_API_KEY in your environment."
        )
    return value


@lru_cache(maxsize=None)
def get_binance_api_secret(*, prefer_testnet: bool = False, required: bool = False) -> str:
    """Return the configured Binance API secret, optionally preferring the testnet variants."""
    candidates: list[str] = []
    if prefer_testnet:
        candidates.extend(_TESTNET_SECRET_VARS)
    candidates.extend(_BINANCE_SECRET_VARS)
    value = _resolve_env_var(candidates)
    if not value and required:
        raise SecretNotConfiguredError(
            "Missing Binance API secret. Set BINANCE_API_SECRET or BINANCE_TESTNET_API_SECRET in your environment."
        )
    return value


def get_api_key(*, required: bool = True) -> str:
    """Backward-compatible helper that returns the live API key."""
    return get_binance_api_key(prefer_testnet=False, required=required)


def get_api_secret(*, required: bool = True) -> str:
    """Backward-compatible helper that returns the live API secret."""
    return get_binance_api_secret(prefer_testnet=False, required=required)


__all__ = [
    "get_binance_api_key",
    "get_binance_api_secret",
    "get_api_key",
    "get_api_secret",
    "SecretNotConfiguredError",
]
