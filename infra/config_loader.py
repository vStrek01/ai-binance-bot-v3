import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from infra.config_schema import AppConfig, RunMode

DEMO_REST_BASE_URL = "https://demo-fapi.binance.com"
DEMO_WS_MARKET_URL = "wss://fstream.binancefuture.com"
DEMO_WS_USER_URL = "wss://fstream.binancefuture.com/ws"
MAINNET_REST_BASE_URL = "https://fapi.binance.com"
MAINNET_WS_MARKET_URL = "wss://fstream.binance.com"
MAINNET_WS_USER_URL = "wss://fstream.binance.com/ws"


class ConfigError(RuntimeError):
    """Raised when configuration or environment validation fails."""


def load_config(path: str = "config.yaml", *, mode_override: str | None = None) -> AppConfig:
    """Load and validate the merged configuration as an AppConfig instance."""

    load_dotenv()
    file_config = _read_config_file(path)
    run_mode = _resolve_run_mode(file_config.get("run_mode"), mode_override)
    merged = _merge_dicts(file_config, _build_env_overrides(run_mode, file_config.get("exchange")))
    _apply_mode_defaults(merged, run_mode)
    merged["run_mode"] = run_mode

    try:
        return AppConfig(**merged)
    except ValidationError as exc:  # pragma: no cover - serialized below
        raise ConfigError(f"Invalid configuration: {exc}") from exc


def _read_config_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigError("config.yaml must contain a mapping at the root level")
        return data


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_run_mode(config_value: Any, override: str | None) -> RunMode:
    candidate = override or os.getenv("RUN_MODE") or config_value or "backtest"
    if not isinstance(candidate, str):
        raise ConfigError("RUN_MODE must be a string if provided")

    normalized = candidate.strip().lower()
    alias_map = {
        "paper": "dry-run",
        "paper-trading": "dry-run",
        "dryrun": "dry-run",
        "demo": "demo-live",
        "demo_live": "demo-live",
        "demolive": "demo-live",
    }
    normalized = alias_map.get(normalized, normalized)

    allowed: tuple[RunMode, ...] = ("backtest", "dry-run", "demo-live", "live")
    if normalized not in allowed:
        allowed_text = ", ".join(allowed)
        raise ConfigError(f"Unsupported RUN_MODE '{candidate}'. Choose one of: {allowed_text}.")

    return normalized  # type: ignore[return-value]


def _build_env_overrides(run_mode: RunMode, exchange_section: Any) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    exchange_override: Dict[str, Any] = {}

    api_key = _strip_env_var("BINANCE_API_KEY")
    api_secret = _strip_env_var("BINANCE_API_SECRET")
    base_url = _strip_env_var("BINANCE_REST_URL") or _strip_env_var("BINANCE_BASE_URL")
    ws_market_url = _strip_env_var("BINANCE_WS_MARKET_URL")
    ws_user_url = _strip_env_var("BINANCE_WS_USER_URL")

    if api_key is not None:
        exchange_override["api_key"] = api_key
    if api_secret is not None:
        exchange_override["api_secret"] = api_secret
    if base_url is not None:
        exchange_override["rest_base_url"] = base_url
    if ws_market_url is not None:
        exchange_override["ws_market_url"] = ws_market_url
    if ws_user_url is not None:
        exchange_override["ws_user_url"] = ws_user_url

    explicit_testnet = os.getenv("BINANCE_TESTNET")
    alias_testnet = os.getenv("BOT_USE_TESTNET")
    raw_testnet = explicit_testnet if explicit_testnet not in {None, ""} else alias_testnet
    parsed_testnet = _parse_optional_bool(raw_testnet)

    exchange_has_testnet = isinstance(exchange_section, dict) and "use_testnet" in exchange_section
    if parsed_testnet is not None:
        exchange_override["use_testnet"] = parsed_testnet
    elif not exchange_has_testnet:
        exchange_override.setdefault("use_testnet", run_mode != "live")

    if exchange_override:
        overrides["exchange"] = exchange_override

    return overrides


def _apply_mode_defaults(config: Dict[str, Any], run_mode: RunMode) -> None:
    exchange = config.setdefault("exchange", {})

    if run_mode == "demo-live":
        exchange.setdefault("use_testnet", True)
        exchange.setdefault("rest_base_url", DEMO_REST_BASE_URL)
        exchange.setdefault("ws_market_url", DEMO_WS_MARKET_URL)
        exchange.setdefault("ws_user_url", DEMO_WS_USER_URL)
    else:
        exchange.setdefault("rest_base_url", MAINNET_REST_BASE_URL)
        exchange.setdefault("ws_market_url", MAINNET_WS_MARKET_URL)
        exchange.setdefault("ws_user_url", MAINNET_WS_USER_URL)

    if run_mode == "live":
        exchange.setdefault("rest_base_url", MAINNET_REST_BASE_URL)
        exchange.setdefault("ws_market_url", MAINNET_WS_MARKET_URL)


def _strip_env_var(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _parse_optional_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError("Boolean env vars must be 1/0 or true/false text")
