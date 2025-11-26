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
    rest_url = _resolve_rest_base_url(cfg)
    if not rest_url:
        raise RuntimeError("Unable to resolve REST base URL for data client. Check exchange.rest_base_url in your config.")
    profile = ClientProfile(
        label="data",
        base_url=rest_url,
        prefer_testnet_keys=cfg.exchange.use_testnet,
        trading=False,
    )
    return ExchangeClient(_build_um_client(profile), mode=profile.label)


def build_trading_client(cfg: BotConfig) -> ExchangeClient:
    """Return an ExchangeClient authorized for order placement."""
    profile = _determine_trading_profile(cfg)
    return ExchangeClient(_build_um_client(profile), mode=profile.label)


def _determine_trading_profile(cfg: BotConfig) -> ClientProfile:
    runtime = cfg.runtime
    if runtime.dry_run:
        raise RuntimeError("Trading client requested while runtime.dry_run=True. Disable dry run to proceed.")
    rest_url = _resolve_rest_base_url(cfg)
    if not rest_url:
        raise RuntimeError("exchange.rest_base_url must be configured before building a trading client")
    if runtime.use_testnet:
        if not cfg.exchange.use_testnet:
            raise RuntimeError("Testnet runtime requested but exchange.use_testnet=False. Update your config to stay on demo endpoints.")
        logger.info("Initializing Binance testnet client", extra={"rest_base": rest_url})
        return ClientProfile(
            label="testnet",
            base_url=rest_url,
            prefer_testnet_keys=True,
            trading=True,
        )
    if runtime.live_trading:
        _ensure_live_confirmation(runtime)
        if cfg.exchange.use_testnet:
            raise RuntimeError("Live trading requested while exchange.use_testnet=True. Configure mainnet endpoints before proceeding.")
        logger.warning("Initializing LIVE Binance client. Proceed with extreme caution.", extra={"rest_base": rest_url})
        return ClientProfile(
            label="live",
            base_url=rest_url,
            prefer_testnet_keys=False,
            trading=True,
        )
    raise RuntimeError("Trading client requested but neither testnet nor live trading is enabled in the runtime config.")


def _resolve_rest_base_url(cfg: BotConfig) -> Optional[str]:
    candidate = (cfg.exchange.rest_base_url or "").strip()
    if candidate:
        return candidate
    runtime = cfg.runtime
    fallback = runtime.testnet_base_url if runtime.use_testnet else runtime.live_base_url
    return fallback.strip() if isinstance(fallback, str) else None


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
