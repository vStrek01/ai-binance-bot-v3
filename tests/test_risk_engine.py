"""Regression tests for the live risk engine guards."""
from __future__ import annotations

from dataclasses import replace

from bot.core.config import BotConfig, load_config
from bot.risk.engine import RiskEngine


def _engine_with(cfg: BotConfig, **overrides: float) -> RiskEngine:
    risk = replace(cfg.risk, **overrides)
    return RiskEngine(replace(cfg, risk=risk))


def test_margin_cap_scales_quantity() -> None:
    cfg = load_config()
    engine = _engine_with(
        cfg,
        margin_buffer=0.05,
        min_free_margin=5.0,
        leverage=20.0,
        max_symbol_exposure=0.0,
        max_account_exposure=0.0,
    )
    qty, reason = engine.adjust_quantity(
        symbol="BTCUSDT",
        price=20_000.0,
        quantity=1.0,
        available_balance=10.0,
        equity=10.0,
        symbol_exposure=0.0,
        total_exposure=0.0,
        filters=None,
    )
    assert qty < 1.0, "quantity should be scaled down when notional exceeds available margin"
    assert reason == "margin"


def test_symbol_cap_blocks_when_exceeded() -> None:
    cfg = load_config()
    engine = _engine_with(cfg, max_symbol_exposure=0.05)
    qty, reason = engine.adjust_quantity(
        symbol="ETHUSDT",
        price=2_000.0,
        quantity=1.0,
        available_balance=1000.0,
        equity=1000.0,
        symbol_exposure=1000.0,  # already at cap
        total_exposure=0.0,
        filters=None,
    )
    assert qty == 0.0
    assert reason == "symbol_cap"


def test_margin_warning_throttled_until_relief() -> None:
    cfg = load_config()
    engine = _engine_with(cfg, margin_warning_cooldown=300.0, margin_relief_factor=1.1)
    first = engine.should_log_margin_block("BTCUSDT", 10.0)
    second = engine.should_log_margin_block("BTCUSDT", 9.0)
    engine.on_balance_refresh(10.5)
    third = engine.should_log_margin_block("BTCUSDT", 12.0)
    assert first is True
    assert second is False, "consecutive warnings should be throttled"
    assert third is True, "warning should be reset once margin recovers"
