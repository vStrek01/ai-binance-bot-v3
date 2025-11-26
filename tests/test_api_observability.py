from __future__ import annotations

from typing import Any, Dict, List

import pytest
from httpx import ASGITransport, AsyncClient

import bot.api as api_module


pytestmark = pytest.mark.anyio


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
async def client() -> AsyncClient:
    transport = ASGITransport(app=api_module.app)
    async with AsyncClient(transport=transport, base_url="http://dashboard.local") as http_client:
        yield http_client


async def test_dashboard_state_endpoint_combines_helpers(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    sample_equity: Dict[str, Any] = {"equity": 10_500, "open_positions": 1, "timestamp": "2024-01-01T00:00:00"}
    sample_positions: List[Dict[str, Any]] = [{"symbol": "BTCUSDT", "side": "LONG", "quantity": 1}]
    sample_events: List[Dict[str, Any]] = [
        {"event": "ORDER_PLACED", "symbol": "BTCUSDT", "timestamp": "2024-01-01T00:00:00"},
        {"event": "KILL_SWITCH_TRIGGERED", "symbol": "BTCUSDT", "timestamp": "2024-01-01T00:01:00"},
    ]
    sample_status = {
        "mode": "demo-live",
        "runtime": {"run_mode": "demo-live", "use_testnet": True, "rest_base_url": "https://demo"},
        "risk_state": {"trading_paused": True, "reason": "daily_loss_pct"},
    }

    monkeypatch.setattr(api_module.logging_utils, "get_equity_snapshot", lambda: sample_equity)
    monkeypatch.setattr(api_module.logging_utils, "get_open_positions", lambda: sample_positions)
    monkeypatch.setattr(api_module.status_store, "snapshot", lambda: sample_status)

    captured_limits: List[int] = []

    def _fake_events(*, limit: int) -> List[Dict[str, Any]]:
        captured_limits.append(limit)
        return sample_events

    monkeypatch.setattr(api_module.logging_utils, "get_recent_events", _fake_events)

    response = await client.get("/api/dashboard/state", params={"limit": 500})
    assert response.status_code == 200
    payload = response.json()
    assert payload["equity"] == sample_equity
    assert payload["positions"] == sample_positions
    assert payload["recent_events"] == sample_events
    assert payload["limit"] == 200  # clamped to MAX_EVENT_LIMIT
    assert payload["run_mode"] == "demo-live"
    assert payload["runtime"]["use_testnet"] is True
    assert payload["risk_state"].get("reason") == "daily_loss_pct"
    assert captured_limits[-1] == 200


async def test_dashboard_equity_endpoint_handles_none(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module.logging_utils, "get_equity_snapshot", lambda: None)
    response = await client.get("/api/dashboard/equity")
    assert response.status_code == 200
    assert response.json() == {"equity": None}


async def test_dashboard_events_endpoint_respects_lower_bound(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_limits: List[int] = []

    def _fake_events(*, limit: int) -> List[Dict[str, Any]]:
        recorded_limits.append(limit)
        return []

    monkeypatch.setattr(api_module.logging_utils, "get_recent_events", _fake_events)

    response = await client.get("/api/dashboard/events", params={"limit": -5})
    assert response.status_code == 200
    assert response.json()["limit"] == 1
    assert recorded_limits[-1] == 1


async def test_dashboard_positions_endpoint_proxy(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    sample_positions = [
        {"symbol": "ETHUSDT", "side": "SHORT", "quantity": 2, "entry_price": 2000.0}
    ]
    monkeypatch.setattr(api_module.logging_utils, "get_open_positions", lambda: sample_positions)
    response = await client.get("/api/dashboard/positions")
    assert response.status_code == 200
    assert response.json() == {"positions": sample_positions}
