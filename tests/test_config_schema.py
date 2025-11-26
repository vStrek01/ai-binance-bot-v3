from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from infra.config_schema import AppConfig, ExchangeConfig, LLMConfig, PathsConfig, RiskConfig, StrategyConfig


def _paths() -> PathsConfig:
    base = Path.cwd()
    return PathsConfig(
        base_dir=base,
        data_dir=base / "data",
        results_dir=base / "results",
        optimization_dir=base / "optimization_results",
        log_dir=base / "logs",
    )


def _build_valid_app_config(**overrides: Any) -> AppConfig:
    payload = {
        "run_mode": "backtest",
        "paths": _paths(),
        "risk": RiskConfig(),
        "exchange": ExchangeConfig(rest_base_url="https://testnet.binancefuture.com"),
        "strategy": StrategyConfig(),
        "llm": LLMConfig(enabled=True, max_confidence=0.8),
    }
    payload.update(overrides)
    return AppConfig(**payload)


def test_valid_config_builds_successfully() -> None:
    cfg = _build_valid_app_config()
    assert cfg.risk.leverage == 3.0
    assert cfg.llm.enabled is True


def test_invalid_run_mode_raises() -> None:
    with pytest.raises(ValidationError):
        AppConfig(run_mode="invalid")


def test_negative_risk_values_raise() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(per_trade_risk=-0.01)


def test_live_mode_without_keys_raises() -> None:
    with pytest.raises(ValidationError, match="api_key"):
        _build_valid_app_config(run_mode="live")


def test_live_mode_with_testnet_enabled_raises() -> None:
    with pytest.raises(ValidationError, match="use_testnet"):
        _build_valid_app_config(
            run_mode="live",
            exchange=ExchangeConfig(
                api_key="k",
                api_secret="s",
                use_testnet=True,
            ),
        )


def test_invalid_strategy_params_raise() -> None:
    with pytest.raises(ValidationError):
        StrategyConfig(parameter_space={"fast_ema": ["invalid"]})


def test_demo_live_requires_testnet_urls() -> None:
    cfg = _build_valid_app_config(
        run_mode="demo-live",
        exchange=ExchangeConfig(
            use_testnet=True,
            rest_base_url="https://demo-fapi.binance.com",
            ws_market_url="wss://fstream.binancefuture.com",
            ws_user_url="wss://fstream.binancefuture.com/ws",
        ),
    )
    assert cfg.exchange.use_testnet is True
    assert "demo" in cfg.exchange.rest_base_url


def test_demo_live_rejects_mainnet_urls() -> None:
    with pytest.raises(ValidationError, match="testnet"):
        _build_valid_app_config(
            run_mode="demo-live",
            exchange=ExchangeConfig(
                use_testnet=True,
                rest_base_url="https://fapi.binance.com",
            ),
        )


def test_live_mode_rejects_demo_urls() -> None:
    with pytest.raises(ValidationError, match="mainnet"):
        _build_valid_app_config(
            run_mode="live",
            exchange=ExchangeConfig(
                api_key="k",
                api_secret="s",
                use_testnet=False,
                rest_base_url="https://demo-fapi.binance.com",
            ),
        )