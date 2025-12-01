from __future__ import annotations

from pathlib import Path

from bot.status import StatusStore


def test_reset_clears_metrics_and_risk(tmp_path: Path) -> None:
    store = StatusStore(persist_path=tmp_path / "status.json")
    store.set_metrics({"profit_factor": 2.0})
    store.set_risk_state({"trading_paused": True, "reason": "kill_switch"})

    snapshot_before = store.snapshot()
    assert snapshot_before["metrics"]["profit_factor"] == 2.0
    assert snapshot_before["risk_state"]["trading_paused"] is True

    store.reset()
    snapshot_after = store.snapshot()
    assert snapshot_after["metrics"] == {}
    assert snapshot_after["risk_state"] == {}
