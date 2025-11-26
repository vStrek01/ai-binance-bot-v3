import importlib
import json
import logging
from io import StringIO
from pathlib import Path

import infra.logging as logging_utils


def _configure_logging(state_file: Path) -> None:
    importlib.reload(logging_utils)
    logging_utils.setup_logging(state_file=str(state_file))


def test_log_event_emits_json_and_updates_equity(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    _configure_logging(state_file)

    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    logging_utils.logger.addHandler(handler)
    try:
        logging_utils.log_event(
            "EQUITY_SNAPSHOT",
            equity=10_500,
            balance=10_500,
            unrealized_pnl=0,
            run_mode="test",
            open_positions=0,
        )
    finally:
        logging_utils.logger.removeHandler(handler)

    handler.flush()
    line = buffer.getvalue().strip().splitlines()[-1]
    json_payload = line[line.find("{") :]
    payload = json.loads(json_payload)
    assert payload["event"] == "EQUITY_SNAPSHOT"
    assert payload["equity"] == 10_500

    dashboard_state = json.loads(state_file.read_text())
    assert dashboard_state["equity"]["equity"] == 10_500


def test_position_events_flow_into_dashboard_state(tmp_path: Path) -> None:
    state_file = tmp_path / "positions.json"
    _configure_logging(state_file)

    logging_utils.log_event(
        "POSITION_OPENED",
        symbol="ETHUSDT",
        side="LONG",
        position={"symbol": "ETHUSDT", "side": "LONG", "quantity": 1},
    )
    data = json.loads(state_file.read_text())
    assert any(pos.get("symbol") == "ETHUSDT" for pos in data.get("positions", []))

    logging_utils.log_event("POSITION_CLOSED", symbol="ETHUSDT", position={"symbol": "ETHUSDT"})
    data = json.loads(state_file.read_text())
    assert all(pos.get("symbol") != "ETHUSDT" for pos in data.get("positions", []))


def test_recent_events_capture_key_event_types(tmp_path: Path) -> None:
    state_file = tmp_path / "events.json"
    _configure_logging(state_file)
    events = [
        ("ORDER_PLACED", {"symbol": "BTCUSDT", "side": "LONG", "qty": 1, "entry_price": 100.0, "run_mode": "test"}),
        ("ORDER_FILLED", {"symbol": "BTCUSDT", "side": "LONG", "fill_price": 101.0, "qty": 1, "status": "FILLED", "run_mode": "test"}),
        (
            "POSITION_OPENED",
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "position": {"symbol": "BTCUSDT", "side": "LONG", "quantity": 1},
            },
        ),
        (
            "POSITION_CLOSED",
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 100.0,
                "exit_price": 102.0,
                "qty": 1,
                "pnl": 2.0,
                "position": {"symbol": "BTCUSDT"},
            },
        ),
        (
            "EQUITY_SNAPSHOT",
            {
                "symbol": "BTCUSDT",
                "mark_price": 102.0,
                "equity": 10_100,
                "balance": 10_100,
                "open_positions": 0,
                "run_mode": "test",
            },
        ),
        (
            "KILL_SWITCH_TRIGGERED",
            {"reason": "drawdown", "equity": 9_000, "daily_start_equity": 10_000, "consecutive_losses": 3, "run_mode": "test"},
        ),
        ("LLM_INVALID_OUTPUT", {"truncated_raw": "", "symbol": "BTCUSDT"}),
        (
            "BACKTEST_SUMMARY",
            {
                "symbol": "BTCUSDT",
                "start": "2024-01-01",
                "end": "2024-01-02",
                "pnl": 123.4,
                "max_drawdown": 0.1,
                "trades": 10,
                "win_rate": 0.6,
                "sharpe": 1.0,
                "strategy_mode": "test",
                "run_mode": "backtest",
                "summary": {"ending_equity": 10_123},
            },
        ),
    ]
    for name, fields in events:
        logging_utils.log_event(name, **fields)

    recorded = [entry["event"] for entry in logging_utils.get_recent_events(limit=len(events))]
    for name, _fields in events:
        assert name in recorded


def test_observability_helpers_return_snapshots(tmp_path: Path) -> None:
    state_file = tmp_path / "observability.json"
    _configure_logging(state_file)
    logging_utils.log_event(
        "EQUITY_SNAPSHOT",
        symbol="SOLUSDT",
        mark_price=50.0,
        equity=12_000.0,
        balance=12_000.0,
        open_positions=1,
        run_mode="test",
    )
    logging_utils.log_event(
        "POSITION_OPENED",
        symbol="SOLUSDT",
        side="LONG",
        position={"symbol": "SOLUSDT", "side": "LONG", "quantity": 2},
    )

    equity_snapshot = logging_utils.get_equity_snapshot()
    assert equity_snapshot is not None
    assert equity_snapshot["equity"] == 12_000.0

    positions = logging_utils.get_open_positions()
    assert any(pos.get("symbol") == "SOLUSDT" for pos in positions)

    recent = logging_utils.get_recent_events(limit=2)
    assert len(recent) == 2
