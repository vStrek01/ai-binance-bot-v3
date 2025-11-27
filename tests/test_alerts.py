from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, cast

import pytest

from infra import alerts


@pytest.fixture(autouse=True)
def reset_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def _noop(url: str, body: bytes, headers: Dict[str, str], timeout: float) -> None:
        del url, body, headers, timeout

    alerts.set_transport(_noop)
    for env in ("ALERT_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(env, raising=False)


def test_webhook_alert_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: List[Tuple[str, Dict[str, object]]] = []

    def fake_transport(url: str, body: bytes, headers: Dict[str, str], timeout: float) -> None:
        captured.append((url, json.loads(body.decode("utf-8"))))

    alerts.set_transport(fake_transport)
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/hook")

    dispatched = alerts.send_alert("TEST_EVENT", severity="info", foo="bar")

    assert dispatched is True
    assert captured[0][0] == "https://example.com/hook"
    payload = cast(Dict[str, Any], captured[0][1])
    assert payload["event"] == "TEST_EVENT"
    assert payload["severity"] == "info"
    details = cast(Dict[str, Any], payload["details"])
    assert details["foo"] == "bar"


def test_telegram_alert_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: List[Tuple[str, Dict[str, object]]] = []

    def fake_transport(url: str, body: bytes, headers: Dict[str, str], timeout: float) -> None:
        captured.append((url, json.loads(body.decode("utf-8"))))

    alerts.set_transport(fake_transport)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")

    dispatched = alerts.send_alert("DATA_UNHEALTHY", severity="warning", message="hello", foo="bar")

    assert dispatched is True
    assert captured[0][0] == "https://api.telegram.org/bot123:abc/sendMessage"
    payload = cast(Dict[str, Any], captured[0][1])
    assert payload["chat_id"] == "999"
    assert "foo=bar" in cast(str, payload["text"])


def test_alert_noop_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel: List[str] = []

    def fake_transport(url: str, body: bytes, headers: Dict[str, str], timeout: float) -> None:
        sentinel.append(url)

    alerts.set_transport(fake_transport)
    dispatched = alerts.send_alert("TEST_EVENT", foo="bar")

    assert dispatched is False
    assert sentinel == []