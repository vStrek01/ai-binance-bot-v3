from __future__ import annotations

import time
from typing import Any, Dict, Tuple, cast

import pytest
from bot.execution import exchange_client as exchange_module
from binance.um_futures import UMFutures  # type: ignore[import-untyped]


class _TimeStub:
    def __init__(self) -> None:
        self.calls = 0

    def time(self):  # type: ignore[override]
        self.calls += 1
        return {"serverTime": int(time.time() * 1000)}


def test_force_time_drift_check_bypasses_cooldown() -> None:
    stub = _TimeStub()
    client = exchange_module.ExchangeClient(cast(UMFutures, stub), mode="test")
    client._last_drift_check = time.time()  # type: ignore[attr-defined]

    client.check_time_drift()
    assert stub.calls == 0  # skipped due to cooldown

    client.check_time_drift(force=True)
    assert stub.calls == 1


def test_client_error_1021_classified_as_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[Tuple[str, Dict[str, Any]]] = []

    def _record(event: str, **fields: Any) -> None:
        events.append((event, fields))

    monkeypatch.setattr(exchange_module, "log_event", _record)

    class _FakeClientError(Exception):
        error_code = -1021
        status_code = 400

    fake = cast(exchange_module.ClientError, _FakeClientError())
    category = exchange_module.ExchangeClient._classify_client_error(fake)

    assert category == "drift"
    assert events and events[0][0] == "TIME_DRIFT_SERVER_REJECTION"


def test_binance_exception_1021_classified_as_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[Tuple[str, Dict[str, Any]]] = []

    def _record(event: str, **fields: Any) -> None:
        events.append((event, fields))

    monkeypatch.setattr(exchange_module, "log_event", _record)

    class _FakeBinanceException(Exception):
        code = -1021
        status_code = 418

    fake = cast(exchange_module.BinanceAPIException, _FakeBinanceException())
    category = exchange_module.ExchangeClient._classify_binance_exception(fake)

    assert category == "drift"
    assert events and events[0][0] == "TIME_DRIFT_SERVER_REJECTION"
