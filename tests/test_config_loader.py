from pathlib import Path

import pytest
from pytest import MonkeyPatch

from infra.config_loader import ConfigError, load_config


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: MonkeyPatch) -> None:
    for key in [
        "RUN_MODE",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "BINANCE_TESTNET",
        "BOT_USE_TESTNET",
        "BINANCE_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_default_run_mode_is_backtest(tmp_path: Path) -> None:
    cfg = load_config(path=str(tmp_path / "missing.yaml"))

    assert cfg.run_mode == "backtest"
    assert cfg.exchange.use_testnet is True


def test_env_aliases_are_normalized(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_MODE", "paper")
    cfg = load_config(path=str(tmp_path / "missing.yaml"))

    assert cfg.run_mode == "dry-run"


def test_mode_override_wins_over_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_MODE", "backtest")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    monkeypatch.setenv("BINANCE_TESTNET", "0")

    cfg = load_config(path=str(tmp_path / "missing.yaml"), mode_override="live")

    assert cfg.run_mode == "live"
    assert cfg.exchange.use_testnet is False


def test_live_mode_requires_credentials(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    with pytest.raises(ConfigError, match=r"exchange\.api_key"):
        load_config(path=str(tmp_path / "missing.yaml"))


def test_live_mode_requires_mainnet(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    monkeypatch.setenv("BINANCE_TESTNET", "1")

    with pytest.raises(ConfigError, match="use_testnet"):
        load_config(path=str(tmp_path / "missing.yaml"))


def test_invalid_testnet_flag_raises(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BINANCE_TESTNET", "not_bool")

    with pytest.raises(ConfigError, match="Boolean env vars"):
        load_config(path=str(tmp_path / "missing.yaml"))


def test_demo_live_defaults_to_testnet_urls(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_MODE", "demo-live")
    cfg = load_config(path=str(tmp_path / "missing.yaml"))

    assert cfg.exchange.use_testnet is True
    assert cfg.exchange.rest_base_url == "https://demo-fapi.binance.com"
    assert "binancefuture" in cfg.exchange.ws_market_url
