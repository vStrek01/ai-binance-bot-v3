"""Shared runtime status store for API and dashboard consumers."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

from bot.utils.fileio import FileLock, atomic_write_text
from bot.utils.logger import get_logger

logger = get_logger(__name__)


def _default_log_dir() -> Path:
    override = os.getenv("BOT_LOG_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "logs"

MAIN_PID_ENV = "STATUS_STORE_MAIN_PID"
if os.getenv(MAIN_PID_ENV) is None:
    os.environ[MAIN_PID_ENV] = str(os.getpid())


class StatusStore:
    """Thread-safe in-memory snapshot of the bot state with disk sync."""

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        *,
        log_dir: Optional[Path] = None,
        default_balance: float = 1_000.0,
    ) -> None:
        resolved_log_dir = (log_dir or _default_log_dir()).resolve()
        target_path = persist_path or (resolved_log_dir / "status.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path: Path = target_path
        self._lock_path: Path = self._persist_path.with_name(self._persist_path.name + ".lock")
        try:
            self._main_pid = int(os.getenv(MAIN_PID_ENV, str(os.getpid())))
        except ValueError:
            self._main_pid = os.getpid()
        self._lock = threading.Lock()
        self._last_loaded_mtime: float = 0.0
        self._log_throttle: Dict[str, float] = {}
        self._default_balance = float(default_balance)
        self._log_dir = resolved_log_dir
        self._state: Dict[str, Any] = self._load_from_disk() or self._default_state()
        if not self._persist_path.exists():
            self._persist_locked()

    def configure(self, *, log_dir: Optional[Path] = None, default_balance: Optional[float] = None) -> None:
        with self._lock:
            if default_balance is not None:
                self._default_balance = float(default_balance)
            if log_dir is not None:
                resolved = log_dir.resolve()
                resolved.mkdir(parents=True, exist_ok=True)
                self._log_dir = resolved
                self._persist_path = resolved / self._persist_path.name
                self._lock_path = self._persist_path.with_name(self._persist_path.name + ".lock")
            if not self._persist_path.exists():
                self._persist_locked()
            else:
                loaded = self._load_from_disk()
                if loaded:
                    self._state = loaded

    def _default_state(self) -> Dict[str, Any]:
        starting_balance = self._default_balance
        return {
            "mode": "idle",
            "run_id": None,
            "run_type": None,
            "symbol": None,
            "timeframe": None,
            "balance": starting_balance,
            "equity": starting_balance,
            "open_pnl": 0.0,
            "realized_pnl": 0.0,
            "open_positions": [],
            "position_index": {},
            "recent_trades": [],
            "symbol_summaries": [],
            "portfolio": {"symbols": []},
            "live_trades": [],
            "live_metrics": {},
            "metrics": {},
            "risk_state": {},
            "rl_state": {
                "active": False,
                "applied": False,
                "context": "disabled",
                "reason": None,
                "symbol": None,
                "timeframe": None,
            },
            "progress": {"completed": 0, "total": 0},
            "updated_at": time.time(),
        }

    def _touch(self) -> None:
        self._state["updated_at"] = time.time()

    def _is_main_process(self) -> bool:
        return os.getpid() == self._main_pid

    def _log_throttled(self, key: str, message: str, level: int = logging.WARNING, interval: float = 1.0) -> None:
        now = time.time()
        last = self._log_throttle.get(key, 0.0)
        if now - last < interval:
            return
        logger.log(level, message)
        self._log_throttle[key] = now

    def _update_last_loaded_mtime(self) -> None:
        if self._persist_path.exists():
            self._last_loaded_mtime = self._persist_path.stat().st_mtime

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._maybe_reload_locked()
            payload = dict(self._state)
            open_positions = self._state.get("open_positions", [])
            if not isinstance(open_positions, list):
                open_positions = []
            payload["open_positions"] = [pos.copy() for pos in open_positions]
            recent_trades = self._state.get("recent_trades", [])
            if not isinstance(recent_trades, list):
                recent_trades = []
            payload["recent_trades"] = [trade.copy() for trade in recent_trades]
            live_trades = self._state.get("live_trades", [])
            if not isinstance(live_trades, list):
                live_trades = []
            payload["live_trades"] = [trade.copy() for trade in live_trades]
            payload["symbol_summaries"] = [entry.copy() for entry in self._state.get("symbol_summaries", [])]
            position_index = self._state.get("position_index", {})
            if not isinstance(position_index, dict):
                position_index = {}
            payload["position_index"] = {key: value.copy() for key, value in position_index.items()}
            portfolio = self._state.get("portfolio", {})
            if not isinstance(portfolio, dict):
                portfolio = {}
            payload["portfolio"] = dict(portfolio)
            live_metrics = self._state.get("live_metrics", {})
            if not isinstance(live_metrics, dict):
                live_metrics = {}
            payload["live_metrics"] = dict(live_metrics)
            risk_state = self._state.get("risk_state", {})
            if not isinstance(risk_state, dict):
                risk_state = {}
            payload["risk_state"] = dict(risk_state)
            rl_state = self._state.get("rl_state", {})
            if not isinstance(rl_state, dict):
                rl_state = {}
            payload["rl_state"] = dict(rl_state)
            progress = self._state.get("progress", {})
            if not isinstance(progress, dict):
                progress = {"completed": 0, "total": 0}
            payload["progress"] = dict(progress)
            payload["run_id"] = self._state.get("run_id")
            payload["run_type"] = self._state.get("run_type")
            return payload

    def set_run_context(self, run_id: str, run_type: Optional[str] = None) -> None:
        with self._lock:
            self._state["run_id"] = run_id
            self._state["run_type"] = run_type or self._state.get("run_type") or "general"
            self._touch()
            self._persist_locked()

    def set_mode(self, mode: str, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        with self._lock:
            self._state["mode"] = mode
            self._state["symbol"] = symbol
            self._state["timeframe"] = timeframe
            self._touch()
            self._persist_locked()

    def update_balance(self, balance: float) -> None:
        with self._lock:
            self._state["balance"] = balance
            self._state["equity"] = balance + self._state.get("open_pnl", 0.0)
            self._touch()
            self._persist_locked()

    def set_open_pnl(self, open_pnl: float) -> None:
        with self._lock:
            self._state["open_pnl"] = open_pnl
            self._state["equity"] = self._state.get("balance", 0.0) + open_pnl
            self._touch()
            self._persist_locked()

    def set_positions(self, positions: List[Dict[str, Any]]) -> None:
        normalized: List[Dict[str, Any]] = []
        index: Dict[str, Dict[str, Any]] = {}
        for entry in positions:
            if not isinstance(entry, dict):
                continue
            quantity = float(entry.get("quantity", 0.0) or 0.0)
            if abs(quantity) <= 0:
                continue
            symbol = str(entry.get("symbol") or "").upper()
            key = f"{symbol}:{entry.get('timeframe') or ''}"
            snapshot = entry.copy()
            snapshot["symbol"] = symbol
            snapshot["quantity"] = quantity
            normalized.append(snapshot)
            index[key] = snapshot
        with self._lock:
            self._state["open_positions"] = normalized
            self._state["position_index"] = index
            self._touch()
            self._persist_locked()

    def add_trade(self, trade: Dict[str, Any], max_items: int = 50) -> None:
        with self._lock:
            trades = [trade.copy()]
            existing = self._state.get("recent_trades", [])
            if isinstance(existing, list):
                trades.extend(existing)
            self._state["recent_trades"] = trades[:max_items]
            pnl = float(trade.get("pnl", 0.0) or 0.0)
            self._state["realized_pnl"] = self._state.get("realized_pnl", 0.0) + pnl
            self._touch()
            self._persist_locked()

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        with self._lock:
            self._state["metrics"] = dict(metrics)
            self._touch()
            self._persist_locked()

    def set_symbol_summaries(self, summaries: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._state["symbol_summaries"] = [entry.copy() for entry in summaries]
            self._touch()
            self._persist_locked()

    def set_portfolio(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._state["portfolio"] = dict(payload)
            self._touch()
            self._persist_locked()

    def add_live_trade(self, trade: Dict[str, Any], max_items: int = 100) -> None:
        with self._lock:
            trades = [trade.copy()]
            existing = self._state.get("live_trades", [])
            if isinstance(existing, list):
                trades.extend(existing)
            self._state["live_trades"] = trades[:max_items]
            self._touch()
            self._persist_locked()

    def set_live_metrics(self, metrics: Dict[str, Any]) -> None:
        with self._lock:
            self._state["live_metrics"] = dict(metrics)
            self._touch()
            self._persist_locked()

    def set_risk_state(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._state["risk_state"] = dict(payload)
            self._touch()
            self._persist_locked()

    def set_rl_state(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._state["rl_state"] = dict(payload)
            self._touch()
            self._persist_locked()

    def integrity_snapshot(self) -> Dict[str, Any]:
        """Return derived metrics useful for lightweight unit checks."""
        with self._lock:
            recent = self._state.get("recent_trades", [])
            realized = sum(float(entry.get("pnl", 0.0) or 0.0) for entry in recent if isinstance(entry, dict))
            summaries = self._state.get("symbol_summaries", [])
            trade_counts = sum(int(entry.get("trades", 0)) for entry in summaries if isinstance(entry, dict))
            return {
                "reported_realized": float(self._state.get("realized_pnl", 0.0) or 0.0),
                "calculated_realized": realized,
                "reported_trade_count": trade_counts,
                "recent_trade_count": len(recent),
            }

    def ledger_is_consistent(self, tolerance: float = 1e-6) -> bool:
        snapshot = self.integrity_snapshot()
        reported = snapshot["reported_realized"]
        calculated = snapshot["calculated_realized"]
        return abs(reported - calculated) <= max(tolerance, abs(reported) * tolerance)

    def set_progress(self, completed: int, total: int) -> None:
        with self._lock:
            self._state["progress"] = {"completed": int(completed), "total": int(total)}
            self._touch()
            self._persist_locked()

    def clear_progress(self) -> None:
        self.set_progress(0, 0)

    def reset(self) -> None:
        with self._lock:
            baseline = self._default_state()
            baseline["updated_at"] = time.time()
            self._state.update(baseline)
            self._persist_locked()

    def _maybe_reload_locked(self) -> None:
        if not self._persist_path.exists():
            return
        mtime = self._persist_path.stat().st_mtime
        if mtime <= self._last_loaded_mtime:
            return
        try:
            payload = json.loads(self._persist_path.read_text())
        except json.JSONDecodeError:
            return
        self._state.update(payload)
        self._last_loaded_mtime = mtime

    def _persist_locked(self) -> None:
        if not self._is_main_process():
            self._log_throttled(
                key="worker_persist_skip",
                message=f"Status persistence skipped in worker process {os.getpid()} (owned by {self._main_pid})",
                level=logging.DEBUG,
                interval=10.0,
            )
            return
        payload = json.dumps(self._state, default=str, indent=2)
        try:
            with FileLock(self._lock_path, timeout=3.0):
                atomic_write_text(self._persist_path, payload)
                self._update_last_loaded_mtime()
        except TimeoutError:
            self._log_throttled("persist_timeout", "Status persistence timed out waiting for lock")
        except OSError as exc:
            self._log_throttled("persist_failure", f"Status persistence failed: {exc}")

    def _load_from_disk(self) -> Optional[Dict[str, Any]]:
        if not self._persist_path.exists():
            return None
        try:
            data = json.loads(self._persist_path.read_text())
        except json.JSONDecodeError:
            return None
        self._update_last_loaded_mtime()
        return data


status_store = StatusStore()

__all__ = ["status_store", "StatusStore"]
