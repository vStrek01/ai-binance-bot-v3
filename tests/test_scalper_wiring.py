from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from bot.execution import runners
from bot.execution.runners import MultiSymbolRunnerBase
from bot.signals.strategies import StrategyParameters, StrategySignal
from infra.config_schema import AppConfig, PathsConfig, ScalpingStrategyConfig
from strategies.ema_stoch_scalping import ScalpingConfig, ScalpingDecision


class DummyExchangeInfo:
    symbols = {"BTCUSDT": {}}

    @staticmethod
    def validate_order(symbol: str, price: float, quantity: float) -> tuple[bool, str | None, float]:
        return True, None, quantity

    @staticmethod
    def get_filters(symbol: str) -> dict:
        return {}


def _paths(tmp_path: Path) -> PathsConfig:
    base = tmp_path
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "results").mkdir(parents=True, exist_ok=True)
    (base / "optimization_results").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    return PathsConfig(
        base_dir=base,
        data_dir=base / "data",
        results_dir=base / "results",
        optimization_dir=base / "optimization_results",
        log_dir=base / "logs",
    )


def test_scalping_adapter_emits_expected_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ScalpingConfig(stop_loss_pct=0.01, take_profit_pct=0.02, size_usd=2_000.0)
    adapter = runners.ScalpingStrategyAdapter(config, symbol="BTCUSDT", interval="1m", run_mode="demo-live")
    frame = pd.DataFrame(
        [
            {
                "open_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "close_time": pd.Timestamp("2024-01-01T00:00:59Z"),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 10.0,
            }
        ]
    )

    decision = ScalpingDecision(
        action="LONG",
        size_usd=config.size_usd,
        sl_pct=config.stop_loss_pct,
        tp_pct=config.take_profit_pct,
        reason="test_long",
        indicators={"ema_fast": 101.0, "ema_slow": 99.0, "stoch_k": 35.0, "stoch_d": 40.0},
    )
    fake_strategy = SimpleNamespace(
        evaluate=lambda candles: decision,
        latest_snapshot={"stoch_k_prev": 12.5},
    )
    adapter._strategy = fake_strategy  # type: ignore[attr-defined]

    signals = adapter.generate_signals(frame)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.direction == 1
    assert signal.stop_loss == pytest.approx(99.0)
    assert signal.take_profit == pytest.approx(102.0)
    assert signal.indicators["size_usd"] == pytest.approx(config.size_usd)
    assert signal.indicators["sl_pct"] == pytest.approx(config.stop_loss_pct)
    assert signal.indicators["tp_pct"] == pytest.approx(config.take_profit_pct)
    assert signal.indicators["stoch_k_prev"] == pytest.approx(12.5)


def test_runner_applies_scalper_size_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = AppConfig(
        paths=_paths(tmp_path),
        symbols=["BTCUSDT"],
        interval="1m",
        scalping_strategy=ScalpingStrategyConfig(size_usd=1_500.0),
        strategy_mode="scalping",
        core_strategy_mode="scalping",
    )
    params = StrategyParameters(
        fast_ema=8,
        slow_ema=21,
        rsi_length=14,
        rsi_overbought=60.0,
        rsi_oversold=40.0,
        atr_period=14,
        atr_stop=1.5,
        atr_target=2.0,
        cooldown_bars=1,
        hold_bars=10,
    )
    exchange = DummyExchangeInfo()
    runner = MultiSymbolRunnerBase(
        markets=[("BTCUSDT", "1m", params)],
        exchange_info=exchange,
        cfg=cfg,
        mode_label="unit-test",
    )
    ctx = runner.contexts[0]
    price = 100.0
    frame = pd.DataFrame(
        [
            {
                "open_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "close_time": pd.Timestamp("2024-01-01T00:00:59Z"),
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 50.0,
            }
        ]
    )
    signal = StrategySignal(
        index=len(frame) - 1,
        direction=1,
        entry_price=price,
        stop_loss=price * 0.998,
        take_profit=price * 1.003,
        indicators={"size_usd": 1_500.0},
        reason="scalper_test",
    )
    ctx.strategy = SimpleNamespace(generate_signals=lambda _: [signal])

    monkeypatch.setattr(runner._risk_engine, "can_open_new_trades", lambda: (True, None))
    monkeypatch.setattr(type(runner._multi_filter), "evaluate", lambda _self, symbol, direction: (True, {}))
    monkeypatch.setattr(runner._signal_gate, "evaluate", lambda symbol, direction: (True, {}))

    expected_qty = signal.indicators["size_usd"] / price
    decision = SimpleNamespace(accepted=True, quantity=expected_qty, reason=None)
    monkeypatch.setattr(runner._sizer, "plan_trade", lambda sizing_ctx, risk_engine: decision)

    runner._maybe_enter(ctx, frame, frame.iloc[-1])

    assert ctx.position is not None
    assert ctx.position.quantity == pytest.approx(expected_qty)
    assert ctx.position.stop_loss == pytest.approx(signal.stop_loss)
    assert ctx.position.take_profit == pytest.approx(signal.take_profit)
    assert ctx.target_notional == pytest.approx(signal.indicators["size_usd"])
    assert runner._max_trade_notional(ctx, price) == pytest.approx(signal.indicators["size_usd"])