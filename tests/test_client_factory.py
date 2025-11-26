from __future__ import annotations

from typing import Any, Dict, List

import pytest
from pytest import MonkeyPatch

from bot.core import config
from bot.execution import client_factory
from bot.execution.client_factory import _ensure_live_confirmation


class _DummyUMFutures:
    last_kwargs: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    def exchange_info(self) -> Dict[str, Any]:  # pragma: no cover - stub
        return {}

    def leverage_bracket(self) -> List[Dict[str, Any]]:  # pragma: no cover - stub
        return []

    def klines(self, **_: Any) -> List[List[Any]]:  # pragma: no cover - stub
        return []


def test_build_trading_client_rejects_dry_run() -> None:
    config.runtime.dry_run = True
    with pytest.raises(RuntimeError):
        client_factory.build_trading_client()


def test_build_trading_client_uses_testnet_profile(monkeypatch: MonkeyPatch) -> None:
    config.runtime.dry_run = False
    config.runtime.use_testnet = True
    config.runtime.live_trading = False

    monkeypatch.setattr(client_factory, "UMFutures", _DummyUMFutures)
    monkeypatch.setattr(client_factory, "get_binance_api_key", lambda prefer_testnet: "test-key")
    monkeypatch.setattr(client_factory, "get_binance_api_secret", lambda prefer_testnet: "test-secret")

    client = client_factory.build_trading_client()
    assert client.mode == "testnet"
    kwargs = _DummyUMFutures.last_kwargs
    assert kwargs["base_url"] == config.runtime.testnet_base_url
    assert kwargs["key"] == "test-key"
    assert kwargs["secret"] == "test-secret"


def test_live_confirmation_env_required(monkeypatch: MonkeyPatch) -> None:
    config.runtime.require_live_confirmation = True
    config.runtime.live_confirmation_env = "BOT_CONFIRM_LIVE"
    monkeypatch.delenv("BOT_CONFIRM_LIVE", raising=False)

    with pytest.raises(RuntimeError):
        _ensure_live_confirmation()

    monkeypatch.setenv("BOT_CONFIRM_LIVE", "1")
    _ensure_live_confirmation()
