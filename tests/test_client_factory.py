from pathlib import Path

import pytest

from bot.core.config import load_config
from bot.execution import client_factory
from bot.execution.exchange_client import ExchangeRequestError


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
    called = {}

    def _fake_verify(client, **_kwargs):
        called["client"] = client

    monkeypatch.setattr(client_factory, "_verify_time_sync", _fake_verify)

    wrapper = client_factory.build_trading_client(cfg)
    assert isinstance(wrapper, _DummyExchangeClient)
    profile = captured["profile"]
    assert profile.label == "testnet"
    assert profile.base_url == cfg.exchange.rest_base_url
    assert profile.prefer_testnet_keys is True
    assert profile.trading is True
    assert called["client"] is wrapper


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
    called = {}

    def _fake_verify(client, **kwargs):
        called["client"] = client

    monkeypatch.setattr(client_factory, "_verify_time_sync", _fake_verify)

    wrapper = client_factory.build_trading_client(cfg)
    assert isinstance(wrapper, _DummyExchangeClient)
    profile = captured["profile"]
    assert profile.label == "live"
    assert profile.base_url == cfg.exchange.rest_base_url
    assert profile.prefer_testnet_keys is False
    assert called["client"] is wrapper


def test_data_client_inherits_exchange_base_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _configured_cfg(tmp_path)
    captured = {}

    def _fake_build(profile: client_factory.ClientProfile):
        captured["profile"] = profile
        return object()

    monkeypatch.setattr(client_factory, "_build_um_client", _fake_build)
    monkeypatch.setattr(client_factory, "ExchangeClient", _DummyExchangeClient)

    def _fail_verify(*_args, **_kwargs):
        raise AssertionError("_verify_time_sync should not run for data clients")

    monkeypatch.setattr(client_factory, "_verify_time_sync", _fail_verify)

    wrapper = client_factory.build_data_client(cfg)
    assert isinstance(wrapper, _DummyExchangeClient)
    profile = captured["profile"]
    assert profile.label == "data"
    assert profile.base_url == cfg.exchange.rest_base_url
    assert profile.trading is False
    assert profile.prefer_testnet_keys is True


def test_verify_time_sync_forces_drift_check(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class _Probe:
        mode = "testnet"

        def check_time_drift(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(client_factory, "log_event", lambda *args, **kwargs: None)

    client_factory._verify_time_sync(_Probe())

    assert calls and calls[0]["force"] is True


def test_verify_time_sync_raises_actionable_error_on_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Probe:
        mode = "testnet"

        def check_time_drift(self, **kwargs):
            raise ExchangeRequestError("server_time", "drift", "Clock drift 1.4s")

    monkeypatch.setattr(client_factory, "log_event", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError) as exc:
        client_factory._verify_time_sync(_Probe())

    assert "Clock drift" in str(exc.value)
