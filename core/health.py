from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from exchange.binance_client import BinanceClient
from exchange.symbols import SymbolResolver
from infra.config_schema import AppConfig
from infra.logging import log_event, logger as base_logger


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_pre_trade_checks(
    config: AppConfig,
    exchange_client: BinanceClient,
    symbol_resolver: SymbolResolver,
    logger=base_logger,
    *,
    run_id: str | None = None,
    ws_client: Any | None = None,
    account_provider: Any | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Run connectivity, market hygiene, and account checks before trading."""

    checks_passed = True
    metadata: Dict[str, Any] = {}
    run_mode = getattr(config, "run_mode", "unknown")

    def record(check: str, status: str, **details: Any) -> None:
        nonlocal checks_passed
        if status != "ok":
            checks_passed = False
        payload = {"check": check, "status": status, **details}
        metadata[check] = payload
        event = "health_check_passed" if status == "ok" else "health_check_failed"
        log_event(event, run_id=run_id, run_mode=run_mode, **payload)

    # Clock drift check
    try:
        server_ms = _safe_int(exchange_client.server_time_ms())
        if server_ms <= 0:
            raise RuntimeError("exchange returned invalid server time")
        local_ms = int(time.time() * 1000)
        drift_ms = abs(server_ms - local_ms)
        allowed = getattr(getattr(config, "health", None), "max_clock_drift_ms", None)
        if allowed is None:
            allowed = getattr(config, "max_clock_drift_ms", 0)
        allowed_ms = allowed or 0
        within_bounds = drift_ms <= allowed_ms if allowed_ms > 0 else True
        record("clock", "ok" if within_bounds else "fail", drift_ms=drift_ms, allowed_ms=allowed_ms)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        record("clock", "error", error=str(exc))
        logger.warning("Clock drift check failed", exc_info=exc)

    # REST connectivity (ping + exchangeInfo)
    try:
        exchange_client.ping()
        exchange_client.exchange_info()
        record("rest", "ok")
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        record("rest", "fail", error=str(exc))
        logger.warning("REST connectivity check failed", exc_info=exc)

    # Websocket connectivity (optional)
    if ws_client is not None:
        try:
            is_connected = False
            if hasattr(ws_client, "is_connected"):
                is_connected = bool(ws_client.is_connected())
            elif hasattr(ws_client, "connected"):
                is_connected = bool(getattr(ws_client, "connected"))
            else:
                raise RuntimeError("ws_client missing status helpers")
            record("websocket", "ok" if is_connected else "fail", connected=is_connected)
        except Exception as exc:  # noqa: BLE001
            record("websocket", "error", error=str(exc))
            logger.warning("Websocket connectivity check failed", exc_info=exc)

    # Symbol universe validation
    try:
        symbol_resolver.refresh()
        snapshot = symbol_resolver.snapshot()
        available = {sym.upper() for sym in snapshot.keys()}
        requested = [sym.upper() for sym in (config.symbols or [])]
        missing = [sym for sym in requested if sym and sym not in available]
        record(
            "symbols",
            "ok" if not missing else "fail",
            missing_symbols=missing,
            total=len(requested),
            available=len(available),
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        record("symbols", "error", error=str(exc))
        logger.warning("Symbol resolver refresh failed", exc_info=exc)

    # Account status / balance
    acct_source = account_provider or exchange_client
    if acct_source is not None:
        try:
            overview = acct_source.account_overview()
            can_trade = bool(overview.get("canTrade", True))
            available = _safe_float(overview.get("availableBalance") or overview.get("maxWithdrawAmount") or 0.0)
            min_balance = getattr(getattr(config, "health", None), "min_available_balance", None)
            if min_balance is None:
                min_balance = getattr(getattr(config, "risk", None), "min_order_notional_usd", 0.0)
            min_balance = _safe_float(min_balance, 0.0)
            meets_balance = available >= (min_balance or 0.0)
            status = "ok" if (can_trade and meets_balance) else "fail"
            record(
                "account",
                status,
                can_trade=can_trade,
                available_balance=available,
                required_balance=min_balance,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            record("account", "error", error=str(exc))
            logger.warning("Account overview check failed", exc_info=exc)

    final_event = "health_check_passed" if checks_passed else "health_check_failed"
    log_event(final_event, run_id=run_id, run_mode=run_mode, summary=True, checks=list(metadata.values()))
    return checks_passed, metadata
