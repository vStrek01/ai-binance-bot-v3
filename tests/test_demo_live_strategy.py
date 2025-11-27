from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

from bot.execution.market_loop import MarketLoop
from bot.execution.runners import MarketContext
from bot.signals.strategies import StrategyParameters, build_parameters
from infra.config_loader import load_config as load_app_config


class _StubRiskGate:
    def __init__(self) -> None:
        self.engine = object()

    def assess_entry(self, _: Any) -> SimpleNamespace:
        return SimpleNamespace(allowed=True, reason=None)


class _StubSizer:
    def __init__(self, quantity: float = 0.01) -> None:
        self.quantity = quantity

    def plan_trade(self, _: Any, __: Any) -> SimpleNamespace:
        return SimpleNamespace(accepted=True, reason=None, quantity=self.quantity)


class _StubFilter:
    def evaluate(self, *_: Any, **__: Any) -> Tuple[bool, Dict[str, float]]:
        return True, {"score": 0.0}


class _StubExchangeInfo:
    def get_filters(self, _: str) -> Dict[str, Any]:
        return {}

    def validate_order(self, _: str, __: float, quantity: float) -> Tuple[bool, str | None, float]:
        return True, None, quantity


def _blank_config(tmp_path: Any, *, mode: str) -> Any:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("", encoding="utf-8")
    return load_app_config(path=str(config_path), base_dir=tmp_path, mode_override=mode)


def _frame(rows: int = 64) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=rows, freq="T")
    data = {
        "open_time": times,
        "close_time": times,
        "open": [100.0 + idx * 0.1 for idx in range(rows)],
        "high": [100.5 + idx * 0.1 for idx in range(rows)],
        "low": [99.5 + idx * 0.1 for idx in range(rows)],
        "close": [100.2 + idx * 0.1 for idx in range(rows)],
        "volume": [1_000.0 for _ in range(rows)],
    }
    return pd.DataFrame(data)


def test_build_parameters_uses_demo_overrides(tmp_path: Any) -> None:
    cfg = _blank_config(tmp_path, mode="demo-live")
    params = build_parameters(cfg)
    overrides = cfg.strategy.demo_live_overrides
    assert params.fast_ema == int(overrides["fast_ema"])
    assert params.slow_ema == int(overrides["slow_ema"])
    assert params.cooldown_bars == int(overrides["cooldown_bars"])
    assert params.hold_bars == int(overrides["hold_bars"])
    assert params.rsi_overbought == pytest.approx(overrides["rsi_overbought"])
    assert params.rsi_oversold == pytest.approx(overrides["rsi_oversold"])


def test_build_parameters_keeps_defaults_for_backtest(tmp_path: Any) -> None:
    cfg = _blank_config(tmp_path, mode="backtest")
    params = build_parameters(cfg)
    defaults = cfg.strategy.default_parameters
    assert params.fast_ema == int(defaults["fast_ema"])
    assert params.slow_ema == int(defaults["slow_ema"])
    assert params.cooldown_bars == int(defaults["cooldown_bars"])
    assert params.hold_bars == int(defaults["hold_bars"])
    assert params.rsi_overbought == pytest.approx(defaults["rsi_overbought"])
    assert params.rsi_oversold == pytest.approx(defaults["rsi_oversold"])


def test_demo_live_strategy_emits_decision(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    cfg = _blank_config(tmp_path, mode="demo-live")
    params: StrategyParameters = build_parameters(cfg)
    events: List[Tuple[str, Dict[str, Any]]] = []

    def _capture(event: str, **fields: Any) -> None:
        events.append((event, fields))

    monkeypatch.setattr("bot.execution.market_loop.log_event", _capture)
    monkeypatch.setattr("bot.signals.strategies.log_event", _capture)

    fast_len = params.fast_ema
    slow_len = params.slow_ema

    def fake_ema(series: pd.Series, length: int) -> pd.Series:
        total = len(series)
        if length == fast_len:
            values = [94.0] * (total - 1) + [105.0]
        else:
            values = [96.0] * total
        return pd.Series(values)

    def fake_rsi(series: pd.Series, __: int) -> pd.Series:
        return pd.Series([params.rsi_oversold - 5.0] * len(series))

    def fake_atr(frame: pd.DataFrame, _: int) -> pd.Series:
        return pd.Series([1.0] * len(frame))

    monkeypatch.setattr("bot.signals.indicators.ema", fake_ema)
    monkeypatch.setattr("bot.signals.indicators.rsi", fake_rsi)
    monkeypatch.setattr("bot.signals.indicators.atr", fake_atr)

    ctx = MarketContext(symbol="BTCUSDT", timeframe="1m", params=params, run_mode="demo-live")
    loop = MarketLoop(
        ctx,
        cfg,
        risk_gate=_StubRiskGate(),
        sizer=_StubSizer(),
        multi_filter=_StubFilter(),
        external_gate=_StubFilter(),
        exchange_info=_StubExchangeInfo(),
        sizing_builder=lambda *_args, **_kwargs: SimpleNamespace(filters={}),
        timestamp_fn=lambda row: str(row.get("close_time")),
        log_sizing_skip=lambda *_args, **_kwargs: None,
    )

    frame = _frame(max(slow_len + 5, 32))
    plan = loop.plan_entry(
        frame,
        frame.iloc[-1],
        balance=1_000.0,
        equity=1_000.0,
        available_balance=1_000.0,
        exposure=SimpleNamespace(),
    )

    assert plan is not None, "Expected an actionable entry plan in demo-live mode"
    assert any(event == "STRATEGY_DECISION" for event, _ in events)
