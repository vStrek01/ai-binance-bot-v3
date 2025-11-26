from __future__ import annotations

import pytest

from exchange import binance_client


@pytest.fixture
def client(monkeypatch):
    instance = binance_client.BinanceClient("key", "secret", base_url="https://demo-fapi.binance.com", testnet=True)
    monkeypatch.setattr(instance, "_timestamp", lambda: 1_000_000)
    return instance


def test_time_drift_warning_logs_event(monkeypatch, client):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(binance_client, "log_event", lambda event, **fields: events.append((event, fields)))
    monkeypatch.setattr(client, "_request", lambda method, path: {"serverTime": 1_000_000 + 3_000})

    client.check_time_drift(warn_threshold=1.0, abort_threshold=5.0)

    assert any(evt == "TIME_DRIFT_WARNING" for evt, _fields in events)


def test_time_drift_abort_raises(monkeypatch, client):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(binance_client, "log_event", lambda event, **fields: events.append((event, fields)))
    monkeypatch.setattr(client, "_request", lambda method, path: {"serverTime": 1_000_000 + 10_000})

    with pytest.raises(RuntimeError):
        client.check_time_drift(warn_threshold=1.0, abort_threshold=2.0)

    assert any(evt == "TIME_DRIFT_CRITICAL" for evt, _fields in events)
