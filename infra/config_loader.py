import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from infra.pydantic_guard import ensure_pydantic_v2
from pydantic import ValidationError

from infra.config_schema import AppConfig, PathsConfig, RunMode
from infra.logging import logger, log_event

ensure_pydantic_v2()

DEMO_REST_BASE_URL = "https://demo-fapi.binance.com"
DEMO_WS_MARKET_URL = "wss://fstream.binancefuture.com"
DEMO_WS_USER_URL = "wss://fstream.binancefuture.com/ws"
MAINNET_REST_BASE_URL = "https://fapi.binance.com"
MAINNET_WS_MARKET_URL = "wss://fstream.binance.com"
MAINNET_WS_USER_URL = "wss://fstream.binance.com/ws"


class ConfigError(RuntimeError):
    """Raised when configuration or environment validation fails."""


def _log_config_snapshot(config: AppConfig, *, source: Dict[str, bool]) -> None:
    sanitized_exchange = config.exchange.model_dump()
    for key in ("api_key", "api_secret"):
        if sanitized_exchange.get(key):
            sanitized_exchange[key] = "***redacted***"
    payload = config.model_dump()
    payload["exchange"] = sanitized_exchange
    log_event(
        "runtime_mode_resolved",
        run_mode=config.run_mode,
        use_testnet=config.exchange.use_testnet,
        rest_base=config.exchange.rest_base_url,
        ws_market=config.exchange.ws_market_url,
        ws_user=config.exchange.ws_user_url,
        source=source,
    )
    log_event("app_config_loaded", config=payload)


def load_config(
    path: str = "config.yaml",
    *,
    base_dir: Path | None = None,
    mode_override: str | None = None,
    cli_overrides: Dict[str, Any] | None = None,
) -> AppConfig:
    """Load and validate the merged configuration as an AppConfig instance."""

    dotenv_path = (base_dir / ".env") if base_dir else None
    load_dotenv(dotenv_path=dotenv_path, override=False)
    file_config = _read_config_file(path)
    run_mode = _resolve_run_mode(file_config.get("run_mode"), mode_override)
    env_overrides = _build_env_overrides(run_mode, file_config.get("exchange"))
    merged = _merge_dicts(file_config, env_overrides)
    if cli_overrides:
        merged = _merge_dicts(merged, cli_overrides)
    _apply_mode_defaults(merged, run_mode)
    merged["run_mode"] = run_mode
    merged["paths"] = _build_paths(base_dir).model_dump()

    try:
        config = AppConfig(**merged)
    except ValidationError as exc:  # pragma: no cover - serialized below
        raise ConfigError(f"Invalid configuration: {exc}") from exc

    _log_config_snapshot(
        config,
        source={
            "file": bool(file_config),
            "env": bool(env_overrides),
            "cli": bool(cli_overrides),
        },
    )
    return config


def load_app_config(
    path: str = "config.yaml",
    *,
    base_dir: Path | None = None,
    mode_override: str | None = None,
    cli_overrides: Dict[str, Any] | None = None,
) -> AppConfig:
    """Public helper exposing the canonical AppConfig loader."""

    return load_config(path=path, base_dir=base_dir, mode_override=mode_override, cli_overrides=cli_overrides)


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
    """Resolve run_mode with CLI > env > YAML precedence."""

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
    runtime_override: Dict[str, Any] = {}

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

    raw_dry_run = _strip_env_var("BOT_DRY_RUN")
    raw_live_trading = _strip_env_var("BOT_LIVE_TRADING")
    raw_use_testnet = _strip_env_var("BOT_USE_TESTNET")

    if raw_dry_run is not None:
        runtime_override["dry_run"] = _parse_optional_bool(raw_dry_run)
    if raw_live_trading is not None:
        runtime_override["live_trading"] = _parse_optional_bool(raw_live_trading)
    if raw_use_testnet is not None:
        runtime_override["use_testnet"] = _parse_optional_bool(raw_use_testnet)

    if runtime_override:
        # Clean None entries if parse_optional_bool returned None
        runtime_override = {k: v for k, v in runtime_override.items() if v is not None}
        if runtime_override:
            overrides.setdefault("runtime", {}).update(runtime_override)

    if exchange_override:
        overrides["exchange"] = exchange_override

    return overrides


def _apply_mode_defaults(config: Dict[str, Any], run_mode: RunMode) -> None:
    runtime = config.setdefault("runtime", {})
    exchange = config.setdefault("exchange", {})
    risk = config.setdefault("risk", {})

    def _ensure_value(target: Dict[str, Any], key: str, value: Any) -> None:
        current = target.get(key)
        if current in {None, ""}:
            target[key] = value

    if run_mode == "demo-live":
        runtime.setdefault("dry_run", False)
        runtime.setdefault("live_trading", True)
        runtime.setdefault("use_testnet", True)
        exchange.setdefault("use_testnet", True)
        _ensure_value(exchange, "rest_base_url", DEMO_REST_BASE_URL)
        _ensure_value(exchange, "ws_market_url", DEMO_WS_MARKET_URL)
        _ensure_value(exchange, "ws_user_url", DEMO_WS_USER_URL)
        risk.setdefault("per_trade_risk", 0.001)
        risk.setdefault("max_daily_loss_pct", 0.10)
        risk.setdefault("max_consecutive_losses", 3)
        risk.setdefault("max_notional_per_symbol", 500.0)
        risk.setdefault("max_notional_global", 1_500.0)
        risk.setdefault("require_sl_tp", True)
    elif run_mode == "live":
        runtime.setdefault("dry_run", False)
        runtime.setdefault("live_trading", True)
        runtime.setdefault("use_testnet", False)
        exchange.setdefault("use_testnet", False)
        exchange.setdefault("rest_base_url", MAINNET_REST_BASE_URL)
        exchange.setdefault("ws_market_url", MAINNET_WS_MARKET_URL)
        exchange.setdefault("ws_user_url", MAINNET_WS_USER_URL)
    else:
        exchange.setdefault("rest_base_url", MAINNET_REST_BASE_URL)
        exchange.setdefault("ws_market_url", MAINNET_WS_MARKET_URL)
        exchange.setdefault("ws_user_url", MAINNET_WS_USER_URL)


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


def _build_paths(base_dir: Path | None) -> PathsConfig:
    candidate = base_dir or _env_path("BOT_BASE_DIR") or Path(__file__).resolve().parents[1]
    candidate = candidate.resolve()
    return PathsConfig(
        base_dir=candidate,
        data_dir=candidate / "data",
        results_dir=candidate / "results",
        optimization_dir=candidate / "optimization_results",
        log_dir=candidate / "logs",
    )


def _env_path(name: str) -> Path | None:
    raw = _strip_env_var(name)
    if raw is None:
        return None
    return Path(raw)
