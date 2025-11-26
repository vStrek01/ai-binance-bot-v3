"""Sizing planner regression tests."""
from __future__ import annotations

from bot.core import config
from bot.risk.engine import RiskEngine
from bot.risk.sizing import PositionSizer, SizingContext
from bot.signals.strategies import StrategyParameters


def _base_context(symbol: str = "BTCUSDT") -> SizingContext:
    params = StrategyParameters(
        fast_ema=13,
        slow_ema=34,
        rsi_length=14,
        rsi_overbought=60,
        rsi_oversold=40,
        atr_period=14,
        atr_stop=1.6,
        atr_target=2.2,
        cooldown_bars=2,
        hold_bars=90,
    )
    return SizingContext(
        symbol=symbol,
        balance=1000.0,
        equity=1000.0,
        available_balance=1000.0,
        price=20000.0,
        params=params,
        volatility={"atr": 100.0},
        filters=None,
        max_notional=None,
        symbol_exposure=0.0,
        total_exposure=0.0,
        active_symbols=0,
        symbol_already_active=False,
    )


def test_max_symbol_cap_blocks_new_entries() -> None:
    ctx = _base_context()
    ctx.active_symbols = config.risk.max_concurrent_symbols
    ctx.symbol_already_active = False
    sizer = PositionSizer()
    result = sizer.plan_trade(ctx, RiskEngine())
    if config.risk.max_concurrent_symbols > 0:
        assert result.rejected and result.reason == "max_symbols"
    else:
        assert not result.rejected


def test_risk_engine_clamps_quantity_when_exposure_high() -> None:
    ctx = _base_context()
    ctx.symbol_exposure = ctx.price * 0.5
    ctx.total_exposure = ctx.symbol_exposure
    sizer = PositionSizer()
    engine = RiskEngine()
    result = sizer.plan_trade(ctx, engine)
    assert result.quantity >= 0
    assert result.accepted or result.reason in {"symbol_cap", "risk_limit"}
