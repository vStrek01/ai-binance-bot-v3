from typing import Any, Dict

import pytest

from bot.core import config
from bot.execution import client_factory


def test_build_trading_client_rejects_dry_run() -> None:
    config.runtime.dry_run = True
    with pytest.raises(RuntimeError):
        client_factory.build_trading_client()


def test_build_trading_client_uses_testnet_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    config.runtime.dry_run = False
    config.runtime.use_testnet = True
    config.runtime.live_trading = False
    captured: Dict[str, Any] = {}

    def fake_builder(profile: client_factory.ClientProfile) -> object:
        captured["profile"] = profile
        return object()

    class FakeExchangeClient:
        def __init__(self, raw: object, mode: str) -> None:
            self.raw = raw
            self.mode = mode

    monkeypatch.setattr(client_factory, "_build_um_client", fake_builder)
    monkeypatch.setattr(client_factory, "ExchangeClient", FakeExchangeClient)

    client = client_factory.build_trading_client()

    assert isinstance(client, FakeExchangeClient)
    profile = captured["profile"]
    assert profile.label == "testnet"
    assert profile.base_url == config.runtime.testnet_base_url
    assert client.mode == "testnet"


def test_live_trading_requires_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    config.runtime.dry_run = False
    config.runtime.use_testnet = False
    config.runtime.live_trading = True
    config.runtime.require_live_confirmation = True
    config.runtime.live_confirmation_env = "BOT_CONFIRM_LIVE"
    monkeypatch.delenv("BOT_CONFIRM_LIVE", raising=False)

    with pytest.raises(RuntimeError):
        client_factory.build_trading_client()

    monkeypatch.setenv("BOT_CONFIRM_LIVE", "1")

    captured: Dict[str, Any] = {}

    def fake_builder(profile: client_factory.ClientProfile) -> object:
        captured["profile"] = profile
        return object()

    class FakeExchangeClient:
        def __init__(self, raw: object, mode: str) -> None:
            self.raw = raw
            self.mode = mode

    monkeypatch.setattr(client_factory, "_build_um_client", fake_builder)
    monkeypatch.setattr(client_factory, "ExchangeClient", FakeExchangeClient)

    client = client_factory.build_trading_client()

    assert isinstance(client, FakeExchangeClient)
    profile = captured["profile"]
    assert profile.label == "live"
    assert profile.base_url == config.runtime.live_base_url
    assert client.mode == "live"
