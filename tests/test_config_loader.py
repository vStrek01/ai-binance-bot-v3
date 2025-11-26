from pathlib import Path

import pytest
from pytest import MonkeyPatch

from infra.config_loader import ConfigError, ConfigLoader


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: MonkeyPatch) -> None:
    for key in [
        "RUN_MODE",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "BINANCE_TESTNET",
        "CONFIRM_LIVE",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_default_run_mode_is_backtest(tmp_path: Path):
    loader = ConfigLoader(path=str(tmp_path / "missing.yaml"))
    config = loader.load()

    assert config["run_mode"] == "backtest"
    assert config["mode_flags"] == {"backtest": True, "paper": False, "live": False}


def test_mode_override_takes_precedence(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("live_trading_enabled: true\n", encoding="utf-8")

    monkeypatch.setenv("RUN_MODE", "paper")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    monkeypatch.setenv("CONFIRM_LIVE", "YES_I_UNDERSTAND_THE_RISK")

    loader = ConfigLoader(path=str(config_path))
    config = loader.load(mode_override="live")

    assert config["run_mode"] == "live"
    assert config["mode_flags"] == {"backtest": False, "paper": False, "live": True}
    assert config["binance"]["testnet"] is False


def test_paper_mode_requires_keys_and_testnet(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    loader = ConfigLoader(path=str(tmp_path / "config.yaml"))

    monkeypatch.setenv("RUN_MODE", "paper")

    with pytest.raises(ConfigError, match="required"):
        loader.load()

    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    monkeypatch.setenv("BINANCE_TESTNET", "0")

    with pytest.raises(ConfigError, match="Paper mode requires BINANCE_TESTNET=1"):
        loader.load()


def test_live_mode_enforces_all_gates(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("live_trading_enabled: true\n", encoding="utf-8")

    loader = ConfigLoader(path=str(config_path))

    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")

    monkeypatch.setenv("BINANCE_TESTNET", "1")
    with pytest.raises(ConfigError, match="Live mode requires BINANCE_TESTNET=0"):
        loader.load()

    monkeypatch.setenv("BINANCE_TESTNET", "0")
    with pytest.raises(ConfigError, match="CONFIRM_LIVE"):
        loader.load()

    monkeypatch.setenv("CONFIRM_LIVE", "YES_I_UNDERSTAND_THE_RISK")
    config_path.write_text("live_trading_enabled: false\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Live trading is disabled"):
        loader.load()

    config_path.write_text("live_trading_enabled: true\n", encoding="utf-8")
    config = loader.load()

    assert config["run_mode"] == "live"
    assert config["binance"]["testnet"] is False
