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
    cfg = load_config(path=str(tmp_path / "missing.yaml"), base_dir=tmp_path)

    assert cfg.run_mode == "backtest"
    assert cfg.exchange.use_testnet is True


def test_env_aliases_are_normalized(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_MODE", "paper")
    cfg = load_config(path=str(tmp_path / "missing.yaml"), base_dir=tmp_path)

    assert cfg.run_mode == "dry-run"


def test_mode_override_wins_over_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_mode: dry-run\n", encoding="utf-8")
    monkeypatch.setenv("RUN_MODE", "backtest")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    monkeypatch.setenv("BINANCE_TESTNET", "0")

    cfg = load_config(path=str(config_path), base_dir=tmp_path, mode_override="live")

    assert cfg.run_mode == "live"
    assert cfg.exchange.use_testnet is False


def test_live_mode_requires_credentials(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    with pytest.raises(ConfigError, match=r"exchange\.api_key"):
        load_config(path=str(tmp_path / "missing.yaml"), base_dir=tmp_path)


def test_live_mode_requires_mainnet(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    monkeypatch.setenv("BINANCE_TESTNET", "1")

    with pytest.raises(ConfigError, match="use_testnet"):
        load_config(path=str(tmp_path / "missing.yaml"), base_dir=tmp_path)


def test_invalid_testnet_flag_raises(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BINANCE_TESTNET", "not_bool")

    with pytest.raises(ConfigError, match="Boolean env vars"):
        load_config(path=str(tmp_path / "missing.yaml"), base_dir=tmp_path)


def test_demo_live_defaults_to_testnet_urls(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUN_MODE", "demo-live")
    cfg = load_config(path=str(tmp_path / "missing.yaml"), base_dir=tmp_path)

    assert cfg.exchange.use_testnet is True
    assert cfg.exchange.rest_base_url == "https://demo-fapi.binance.com"
    assert "binancefuture" in cfg.exchange.ws_market_url


def test_yaml_run_mode_without_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_mode: demo-live\n", encoding="utf-8")

    cfg = load_config(path=str(config_path), base_dir=tmp_path)

    assert cfg.run_mode == "demo-live"
    assert cfg.exchange.use_testnet is True


def test_env_run_mode_beats_yaml(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_mode: backtest\n", encoding="utf-8")
    monkeypatch.setenv("RUN_MODE", "demo-live")

    cfg = load_config(path=str(config_path), base_dir=tmp_path)

    assert cfg.run_mode == "demo-live"


def test_demo_live_runtime_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_mode: demo-live\n", encoding="utf-8")

    cfg = load_config(path=str(config_path), base_dir=tmp_path)

    assert cfg.runtime.dry_run is False
    assert cfg.runtime.live_trading is True
    assert cfg.runtime.use_testnet is True


def test_demo_live_risk_daily_loss_cap_is_ten_percent(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_mode: demo-live\n", encoding="utf-8")

    cfg = load_config(path=str(config_path), base_dir=tmp_path)

    assert cfg.risk.max_daily_loss_pct == pytest.approx(0.10)


def test_non_demo_modes_keep_existing_daily_loss_cap(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_mode: backtest
risk:
  max_daily_loss_pct: 0.01
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(path=str(config_path), base_dir=tmp_path)

    assert cfg.risk.max_daily_loss_pct == pytest.approx(0.01)
