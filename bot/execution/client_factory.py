"""Centralized Binance client creation with environment safety checks."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from binance.um_futures import UMFutures  # type: ignore[import-untyped]

from bot.core.config import BotConfig, RuntimeConfig
from bot.core.secrets import SecretNotConfiguredError, get_binance_api_key, get_binance_api_secret
from bot.execution.exchange_client import ExchangeClient
from bot.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ClientProfile:
    label: str
    base_url: Optional[str]
    prefer_testnet_keys: bool
    trading: bool


def build_data_client(cfg: BotConfig) -> ExchangeClient:
    """Return an ExchangeClient suited for read-only operations."""
    profile = ClientProfile(label="data", base_url=None, prefer_testnet_keys=False, trading=False)
    return ExchangeClient(_build_um_client(profile), mode=profile.label)


def build_trading_client(cfg: BotConfig) -> ExchangeClient:
    """Return an ExchangeClient authorized for order placement."""
    profile = _determine_trading_profile(cfg)
    return ExchangeClient(_build_um_client(profile), mode=profile.label)


def _determine_trading_profile(cfg: BotConfig) -> ClientProfile:
    runtime = cfg.runtime
    if runtime.dry_run:
        raise RuntimeError("Trading client requested while runtime.dry_run=True. Disable dry run to proceed.")
    if runtime.use_testnet:
        logger.info("Initializing Binance testnet client")
        return ClientProfile(
            label="testnet",
            base_url=runtime.testnet_base_url,
            prefer_testnet_keys=True,
            trading=True,
        )
    if runtime.live_trading:
        _ensure_live_confirmation(runtime)
        logger.warning("Initializing LIVE Binance client. Proceed with extreme caution.")
        return ClientProfile(
            label="live",
            base_url=runtime.live_base_url,
            prefer_testnet_keys=False,
            trading=True,
        )
    raise RuntimeError("Trading client requested but neither testnet nor live trading is enabled in the runtime config.")


def _ensure_live_confirmation(runtime: RuntimeConfig) -> None:
    if not runtime.require_live_confirmation:
        return
    env_flag = runtime.live_confirmation_env or "BOT_CONFIRM_LIVE"
    if os.getenv(env_flag) not in {"1", "true", "TRUE", "yes", "YES"}:
        raise RuntimeError(
            f"Live trading requires explicit acknowledgement via {env_flag}=1. Set the variable and retry."
        )


def _build_um_client(profile: ClientProfile) -> UMFutures:
    key = get_binance_api_key(prefer_testnet=profile.prefer_testnet_keys)
    secret = get_binance_api_secret(prefer_testnet=profile.prefer_testnet_keys)
    kwargs: Dict[str, Any] = {}
    if profile.base_url:
        kwargs["base_url"] = profile.base_url
    if key and secret:
        kwargs.update({"key": key, "secret": secret})
    elif profile.trading:
        raise SecretNotConfiguredError(
            "Trading client requires Binance API credentials. Export BINANCE_API_KEY/BINANCE_API_SECRET first."
        )
    return UMFutures(**kwargs)


__all__ = ["build_data_client", "build_trading_client", "ClientProfile"]
