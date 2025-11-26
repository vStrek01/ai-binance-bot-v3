from pathlib import Path

import pytest

from bot.core.config import load_config
from bot.execution import client_factory


class _DummyExchangeClient:
    def __init__(self, raw_client, mode: str):
        self.raw = raw_client
        self.mode = mode


def _configured_cfg(tmp_path: Path):
    cfg = load_config(base_dir=tmp_path)
    runtime = cfg.runtime.model_copy(update={"dry_run": False, "live_trading": True})
    exchange = cfg.exchange.model_copy(update={"rest_base_url": "https://demo-fapi.binance.com", "use_testnet": True})
    return cfg.model_copy(update={"runtime": runtime, "exchange": exchange})


def test_trading_client_uses_exchange_rest_url_for_testnet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _configured_cfg(tmp_path)
    runtime = cfg.runtime.model_copy(update={"use_testnet": True})
    cfg = cfg.model_copy(update={"runtime": runtime})
    captured = {}

    def _fake_build(profile: client_factory.ClientProfile):
        captured["profile"] = profile
        return object()

    monkeypatch.setattr(client_factory, "_build_um_client", _fake_build)
    monkeypatch.setattr(client_factory, "ExchangeClient", _DummyExchangeClient)

    wrapper = client_factory.build_trading_client(cfg)
    assert isinstance(wrapper, _DummyExchangeClient)
    profile = captured["profile"]
    assert profile.label == "testnet"
    assert profile.base_url == cfg.exchange.rest_base_url
    assert profile.prefer_testnet_keys is True
    assert profile.trading is True


def test_trading_client_uses_exchange_rest_url_for_live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _configured_cfg(tmp_path)
    runtime = cfg.runtime.model_copy(update={"use_testnet": False})
    exchange = cfg.exchange.model_copy(update={"rest_base_url": "https://fapi.binance.com", "use_testnet": False})
    cfg = cfg.model_copy(update={"runtime": runtime, "exchange": exchange})
    monkeypatch.setenv(cfg.runtime.live_confirmation_env, "1")
    captured = {}

    def _fake_build(profile: client_factory.ClientProfile):
        captured["profile"] = profile
        return object()

    monkeypatch.setattr(client_factory, "_build_um_client", _fake_build)
    monkeypatch.setattr(client_factory, "ExchangeClient", _DummyExchangeClient)

    wrapper = client_factory.build_trading_client(cfg)
    assert isinstance(wrapper, _DummyExchangeClient)
    profile = captured["profile"]
    assert profile.label == "live"
    assert profile.base_url == cfg.exchange.rest_base_url
    assert profile.prefer_testnet_keys is False


def test_data_client_inherits_exchange_base_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _configured_cfg(tmp_path)
    captured = {}

    def _fake_build(profile: client_factory.ClientProfile):
        captured["profile"] = profile
        return object()

    monkeypatch.setattr(client_factory, "_build_um_client", _fake_build)
    monkeypatch.setattr(client_factory, "ExchangeClient", _DummyExchangeClient)

    wrapper = client_factory.build_data_client(cfg)
    assert isinstance(wrapper, _DummyExchangeClient)
    profile = captured["profile"]
    assert profile.label == "data"
    assert profile.base_url == cfg.exchange.rest_base_url
    assert profile.trading is False
    assert profile.prefer_testnet_keys is True
