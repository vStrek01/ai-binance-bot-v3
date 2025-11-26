from typing import Any

import pytest
from pydantic import ValidationError

from infra.config_schema import AppConfig, ExchangeConfig, LLMConfig, RiskConfig, StrategyConfig


def _build_valid_app_config(**overrides: Any) -> AppConfig:
    payload = {
        "run_mode": "backtest",
        "risk": RiskConfig(
            risk_per_trade_pct=1.5,
            max_daily_drawdown_pct=6.0,
            max_consecutive_losses=4,
            max_notional_per_symbol=2_500.0,
            max_notional_global=10_000.0,
        ),
        "exchange": ExchangeConfig(rest_base_url="https://testnet.binancefuture.com"),
        "strategy": StrategyConfig(
            ema_fast=8,
            ema_slow=34,
            rsi_length=14,
            rsi_overbought=65,
            rsi_oversold=35,
            atr_length=14,
            atr_multiplier=1.8,
        ),
        "llm": LLMConfig(enabled=True, max_confidence=0.8),
    }
    payload.update(overrides)
    return AppConfig(**payload)


def test_valid_config_builds_successfully() -> None:
    cfg = _build_valid_app_config()
    assert cfg.risk.max_notional_global == 10_000.0
    assert cfg.llm.enabled is True


def test_invalid_run_mode_raises() -> None:
    with pytest.raises(ValidationError):
        AppConfig(run_mode="invalid")


def test_negative_risk_values_raise() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(
            risk_per_trade_pct=-1.0,
            max_daily_drawdown_pct=5.0,
            max_consecutive_losses=3,
            max_notional_per_symbol=1_000.0,
            max_notional_global=5_000.0,
        )


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
        StrategyConfig(
            ema_fast=0,
            ema_slow=34,
            rsi_length=14,
            rsi_overbought=60,
            rsi_oversold=40,
            atr_length=14,
            atr_multiplier=1.5,
        )


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