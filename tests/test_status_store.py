"""Status store integrity regression tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

from bot.status import StatusStore


def _make_store(tmp_path: Path) -> StatusStore:
    target = tmp_path / "status.json"
    return StatusStore(persist_path=target)


def test_position_snapshot_and_index_consistent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(Path(tmp))
        store.set_positions(
            [
                {"symbol": "BTCUSDT", "timeframe": "1m", "quantity": 0.5, "entry_price": 20000.0, "mark_price": 20100.0},
                {"symbol": "ETHUSDT", "timeframe": "1m", "quantity": 1.0, "entry_price": 1500.0, "mark_price": 1510.0},
            ]
        )
        snapshot = store.snapshot()
        assert len(snapshot["open_positions"]) == 2
        assert "BTCUSDT:1m" in snapshot["position_index"]


def test_integrity_snapshot_matches_realized_pnl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(Path(tmp))
        store.add_trade({"symbol": "BTCUSDT", "pnl": 10.0})
        store.add_trade({"symbol": "ETHUSDT", "pnl": -3.0})
        store.set_symbol_summaries(
            [
                {"symbol": "BTCUSDT", "timeframe": "1m", "trades": 1},
                {"symbol": "ETHUSDT", "timeframe": "1m", "trades": 1},
            ]
        )
        snapshot = store.integrity_snapshot()
        assert snapshot["reported_realized"] == snapshot["calculated_realized"]
        assert store.ledger_is_consistent()