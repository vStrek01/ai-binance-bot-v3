from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from bot.crash_recovery import (
    bootstrap_crash_recovery,
    compare_positions,
    load_live_positions,
    parse_exchange_positions,
)
from bot.crash_recovery import CrashPosition
from bot.execution.exchange_client import ExchangeClient
from infra.state_store import StateStore


def test_load_live_positions_extracts_directions() -> None:
    snapshot: Dict[str, Any] = {
        "live_positions": {
            "positions": {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "timeframe": "1m",
                    "side": "LONG",
                    "direction": 1,
                    "quantity": 0.25,
                },
                "ETHUSDT": {
                    "symbol": "ETHUSDT",
                    "timeframe": "1m",
                    "side": "SHORT",
                    "quantity": 1.5,
                },
            }
        }
    }
    result = load_live_positions(snapshot)
    assert result["BTCUSDT"].direction == 1
    assert result["ETHUSDT"].direction == -1
    assert result["ETHUSDT"].quantity == 1.5


def test_parse_exchange_positions_combines_entries() -> None:
    entries = [
        {"symbol": "BTCUSDT", "positionAmt": "0.4"},
        {"symbol": "BTCUSDT", "positionAmt": "-0.1"},
        {"symbol": "ETHUSDT", "positionAmt": "-1.0"},
    ]
    result = parse_exchange_positions(entries)
    assert result["BTCUSDT"].direction == 1
    assert result["BTCUSDT"].quantity == pytest.approx(0.3)
    assert result["ETHUSDT"].direction == -1
    assert result["ETHUSDT"].quantity == 1.0


def test_compare_positions_flags_mismatches() -> None:
    expected = {
        "BTCUSDT": CrashPosition(symbol="BTCUSDT", direction=1, quantity=0.5, timeframe="1m"),
        "ETHUSDT": CrashPosition(symbol="ETHUSDT", direction=-1, quantity=1.0, timeframe="1m"),
    }
    actual = {
        "BTCUSDT": CrashPosition(symbol="BTCUSDT", direction=-1, quantity=0.5, timeframe="1m"),
        "SOLUSDT": CrashPosition(symbol="SOLUSDT", direction=1, quantity=0.2, timeframe="1m"),
    }
    report = compare_positions(resume_run_id="run-prev", expected=expected, actual=actual)
    assert report.diverged is True
    assert report.missing == ["ETHUSDT"]
    assert report.unexpected == ["SOLUSDT"]
    assert report.mismatched[0]["symbol"] == "BTCUSDT"


class _StubClient:
    def __init__(self, payload: List[Dict[str, str]]):
        self._payload = payload

    def get_position_risk(self) -> List[Dict[str, str]]:
        return self._payload


def test_bootstrap_crash_recovery_records_summary(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    previous_store = StateStore("run-prev", base_dir=state_dir)
    previous_store.merge(
        live_positions={
            "positions": {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "timeframe": "1m",
                    "side": "LONG",
                    "quantity": 0.5,
                }
            }
        }
    )
    client = cast(ExchangeClient, _StubClient([{"symbol": "BTCUSDT", "positionAmt": "0.25"}]))
    current_store = StateStore("run-new", base_dir=state_dir)
    report = bootstrap_crash_recovery(
        base_dir=state_dir,
        resume_run_id="run-prev",
        trading_client=client,
        state_store=current_store,
    )
    assert report is not None
    snapshot = current_store.load()
    recovery_section = snapshot.get("recovery")
    assert isinstance(recovery_section, dict)
    assert recovery_section["resume_run_id"] == "run-prev"
    assert recovery_section["diverged"] is True