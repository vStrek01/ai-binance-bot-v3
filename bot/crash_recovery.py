"""Helpers for reconciling state after an unplanned shutdown."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from bot.execution.exchange_client import ExchangeClient, ExchangeRequestError
from infra.alerts import send_alert
from infra.logging import log_event
from infra.state_store import StateStore


@dataclass(slots=True)
class CrashPosition:
    symbol: str
    direction: int
    quantity: float
    timeframe: str | None = None


@dataclass(slots=True)
class CrashRecoveryReport:
    resume_run_id: str
    diverged: bool
    expected_symbols: list[str]
    actual_symbols: list[str]
    missing: list[str]
    unexpected: list[str]
    mismatched: list[Dict[str, Any]]


def load_live_positions(snapshot: Mapping[str, Any]) -> Dict[str, CrashPosition]:
    section = snapshot.get("live_positions")
    if not isinstance(section, Mapping):
        return {}
    positions_obj = section.get("positions")
    if not isinstance(positions_obj, Mapping):
        return {}
    result: Dict[str, CrashPosition] = {}
    for key, raw in positions_obj.items():
        if not isinstance(raw, Mapping):
            continue
        symbol = str(raw.get("symbol") or key or "").upper()
        if not symbol:
            continue
        raw_quantity = _safe_float(raw.get("quantity"), 0.0)
        quantity = abs(raw_quantity)
        if quantity <= 0:
            continue
        direction_raw = raw.get("direction")
        direction = _resolve_direction(direction_raw, raw.get("side"), raw_quantity)
        timeframe = raw.get("timeframe")
        result[symbol] = CrashPosition(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            timeframe=str(timeframe) if isinstance(timeframe, str) and timeframe else None,
        )
    return result


def parse_exchange_positions(entries: Optional[Iterable[Mapping[str, Any]]]) -> Dict[str, CrashPosition]:
    result: Dict[str, CrashPosition] = {}
    if entries is None:
        return result
    for entry in entries:
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        qty = _safe_float(entry.get("positionAmt"), 0.0)
        if abs(qty) <= 0.0:
            continue
        direction = 1 if qty > 0 else -1
        quantity = abs(qty)
        existing = result.get(symbol)
        if existing:
            net_signed = existing.direction * existing.quantity + (direction * quantity)
            if abs(net_signed) <= 0.0:
                result.pop(symbol, None)
                continue
            direction = 1 if net_signed > 0 else -1
            quantity = abs(net_signed)
        result[symbol] = CrashPosition(symbol=symbol, direction=direction, quantity=quantity)
    return result


def compare_positions(
    *,
    resume_run_id: str,
    expected: Dict[str, CrashPosition],
    actual: Dict[str, CrashPosition],
    tolerance: float = 1e-6,
) -> CrashRecoveryReport:
    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    mismatched: list[Dict[str, Any]] = []
    for symbol in sorted(expected_keys & actual_keys):
        exp = expected[symbol]
        act = actual[symbol]
        qty_diff = abs(exp.quantity - act.quantity)
        if exp.direction != act.direction or qty_diff > max(tolerance, exp.quantity * tolerance):
            mismatched.append(
                {
                    "symbol": symbol,
                    "expected_qty": exp.quantity,
                    "actual_qty": act.quantity,
                    "expected_direction": exp.direction,
                    "actual_direction": act.direction,
                }
            )
    diverged = bool(missing or unexpected or mismatched)
    return CrashRecoveryReport(
        resume_run_id=resume_run_id,
        diverged=diverged,
        expected_symbols=sorted(expected_keys),
        actual_symbols=sorted(actual_keys),
        missing=missing,
        unexpected=unexpected,
        mismatched=mismatched,
    )


def log_recovery_report(report: CrashRecoveryReport) -> None:
    if report.diverged:
        payload = {
            "resume_run_id": report.resume_run_id,
            "missing": report.missing,
            "unexpected": report.unexpected,
            "mismatched": report.mismatched,
            "expected_symbols": report.expected_symbols,
            "actual_symbols": report.actual_symbols,
        }
        log_event("STATE_DIVERGENCE", **payload)
        send_alert(
            "STATE_DIVERGENCE",
            severity="critical",
            message="Crash recovery detected live divergence",
            **payload,
        )
    else:
        log_event(
            "STATE_RECOVERY_ALIGNED",
            resume_run_id=report.resume_run_id,
            symbols=report.expected_symbols,
        )


def persist_recovery_summary(state_store: StateStore | None, report: CrashRecoveryReport) -> None:
    if state_store is None:
        return
    payload = {
        "resume_run_id": report.resume_run_id,
        "diverged": report.diverged,
        "missing": report.missing,
        "unexpected": report.unexpected,
        "mismatched": report.mismatched,
        "expected_symbols": report.expected_symbols,
        "actual_symbols": report.actual_symbols,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        state_store.merge(recovery=payload)
    except Exception as exc:  # pragma: no cover - persistence is best-effort
        log_event("STATE_RECOVERY_PERSIST_FAILED", resume_run_id=report.resume_run_id, error=str(exc))


def bootstrap_crash_recovery(
    *,
    base_dir: Path,
    resume_run_id: str,
    trading_client: ExchangeClient,
    state_store: StateStore | None,
    tolerance: float = 1e-6,
) -> CrashRecoveryReport | None:
    snapshot = _load_previous_snapshot(base_dir, resume_run_id)
    if snapshot is None:
        log_event("STATE_RECOVERY_SKIPPED", resume_run_id=resume_run_id, reason="missing_snapshot")
        return None
    expected = load_live_positions(snapshot)
    try:
        payload = trading_client.get_position_risk()
    except ExchangeRequestError as exc:
        log_event(
            "STATE_RECOVERY_SKIPPED",
            resume_run_id=resume_run_id,
            reason="position_risk_failed",
            category=exc.category,
        )
        return None
    actual = parse_exchange_positions(payload)
    report = compare_positions(
        resume_run_id=resume_run_id,
        expected=expected,
        actual=actual,
        tolerance=tolerance,
    )
    log_recovery_report(report)
    persist_recovery_summary(state_store, report)
    return report


def _load_previous_snapshot(base_dir: Path, run_id: str) -> Dict[str, Any]:
    store = StateStore(run_id, base_dir=base_dir)
    return store.load()


def _safe_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return fallback
    return parsed


def _resolve_direction(direction: Any, side: Any, quantity: float) -> int:
    if isinstance(direction, (int, float)) and direction != 0:
        return 1 if direction > 0 else -1
    side_str = str(side or "").upper()
    if side_str in {"LONG", "BUY"}:
        return 1
    if side_str in {"SHORT", "SELL"}:
        return -1
    return 1 if quantity >= 0 else -1


__all__ = [
    "CrashPosition",
    "CrashRecoveryReport",
    "load_live_positions",
    "parse_exchange_positions",
    "compare_positions",
    "log_recovery_report",
    "persist_recovery_summary",
    "bootstrap_crash_recovery",
]
