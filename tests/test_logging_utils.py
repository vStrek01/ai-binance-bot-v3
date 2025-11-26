import json
import logging
from io import StringIO
from pathlib import Path

from infra.logging import log_event, logger, setup_logging


def test_log_event_emits_json_and_updates_equity(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    setup_logging(state_file=str(state_file))

    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    logger.addHandler(handler)
    try:
        log_event(
            "EQUITY_SNAPSHOT",
            equity=10_500,
            balance=10_500,
            unrealized_pnl=0,
            run_mode="test",
            open_positions=0,
        )
    finally:
        logger.removeHandler(handler)

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
    setup_logging(state_file=str(state_file))

    log_event(
        "POSITION_OPENED",
        symbol="ETHUSDT",
        side="LONG",
        position={"symbol": "ETHUSDT", "side": "LONG", "quantity": 1},
    )
    data = json.loads(state_file.read_text())
    assert any(pos.get("symbol") == "ETHUSDT" for pos in data.get("positions", []))

    log_event("POSITION_CLOSED", symbol="ETHUSDT", position={"symbol": "ETHUSDT"})
    data = json.loads(state_file.read_text())
    assert all(pos.get("symbol") != "ETHUSDT" for pos in data.get("positions", []))
