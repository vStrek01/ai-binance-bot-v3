from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.models import Candle, Side
from strategies.ema_stoch_scalping import EMAStochasticStrategy, ScalpingConfig, ScalpingParams


def _build_candles(prices: list[float]) -> list[Candle]:
    base = datetime(2024, 1, 1)
    candles: list[Candle] = []
    for idx, price in enumerate(prices):
        open_time = base + timedelta(minutes=idx)
        close_time = open_time + timedelta(seconds=59)
        candles.append(
            Candle(
                symbol="BTCUSDT",
                open_time=open_time,
                close_time=close_time,
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=100 + idx,
            )
        )
    return candles


def test_scalping_strategy_defaults_to_hyper_aggressive_params() -> None:
    strategy = EMAStochasticStrategy()
    params = strategy.params
    assert params.preset == "HYPER_AGGRESSIVE"
    assert params.long_oversold_k == pytest.approx(45.0)
    assert params.short_overbought_k == pytest.approx(55.0)
    assert params.long_cross_min_k == pytest.approx(0.0)
    assert params.short_cross_max_k == pytest.approx(100.0)
    assert params.min_bars_between_trades == 0


def test_scalping_aggressive_preset_differs_from_hyper_defaults() -> None:
    aggressive = ScalpingParams.aggressive()
    assert aggressive.preset == "AGGRESSIVE"
    assert aggressive.long_oversold_k == pytest.approx(40.0)
    assert aggressive.short_overbought_k == pytest.approx(60.0)
    assert aggressive.min_bars_between_trades == 1


def test_scalping_strategy_triggers_long_on_oversold_uptrend(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3, size_usd=500)
    strategy = EMAStochasticStrategy(config)
    candles = _build_candles([100, 101, 102, 103, 104])
    idx_prev = len(candles) - 2
    idx_now = len(candles) - 1

    def fake_stoch_k(highs, lows, closes, idx):  # noqa: ANN001 - signature matches target method
        return 30.0 if idx == idx_now else 45.0

    def fake_stoch_d(highs, lows, closes, idx):  # noqa: ANN001 - signature matches target method
        return 40.0 if idx == idx_now else 50.0

    def fake_ema(values, length):  # noqa: ANN001
        return 105.0 if length == strategy.config.fast_ema else 100.0

    monkeypatch.setattr(strategy, "_stoch_k", fake_stoch_k)
    monkeypatch.setattr(strategy, "_stoch_d", fake_stoch_d)
    monkeypatch.setattr(strategy, "_ema_latest", fake_ema)

    decision = strategy.evaluate(candles)

    assert decision.action == "LONG"
    assert decision.reason == "long_trend_oversold"
    assert decision.size_usd == pytest.approx(500)
    assert strategy.latest_snapshot is not None
    assert strategy.latest_snapshot.get("position_state") == Side.FLAT.value


def test_scalping_strategy_respects_custom_params(monkeypatch: pytest.MonkeyPatch) -> None:
    params = ScalpingParams(
        preset="CUSTOM",
        long_oversold_k=10.0,
        short_overbought_k=90.0,
        long_cross_min_k=80.0,
        short_cross_max_k=95.0,
        min_bars_between_trades=1,
    )
    config = ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3, params=params)
    strategy = EMAStochasticStrategy(config)
    candles = _build_candles([100, 101, 102, 103])

    def fake_stoch_k(highs, lows, closes, idx):  # noqa: ANN001
        current_idx = len(highs) - 1
        if idx == current_idx:
            return 30.0
        return 20.0

    def fake_stoch_d(highs, lows, closes, idx):  # noqa: ANN001
        return 15.0

    def fake_ema(values, length):  # noqa: ANN001
        return 110.0 if length == config.fast_ema else 90.0

    monkeypatch.setattr(strategy, "_stoch_k", fake_stoch_k)
    monkeypatch.setattr(strategy, "_stoch_d", fake_stoch_d)
    monkeypatch.setattr(strategy, "_ema_latest", fake_ema)

    decision = strategy.evaluate(candles)

    assert decision.action == "FLAT"
    assert decision.reason == "no_signal"


def test_hyper_aggressive_ignores_trend_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    hyper_strategy = EMAStochasticStrategy(
        ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3, params=ScalpingParams.hyper_aggressive())
    )
    aggressive_strategy = EMAStochasticStrategy(
        ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3, params=ScalpingParams.aggressive())
    )
    candles = _build_candles([100, 100.5, 100.25, 100.1])

    def flat_ema(values, length):  # noqa: ANN001
        return 100.0

    def oversold_k(highs, lows, closes, idx):  # noqa: ANN001
        return 30.0

    def oversold_d(highs, lows, closes, idx):  # noqa: ANN001
        return 40.0

    for strat in (hyper_strategy, aggressive_strategy):
        monkeypatch.setattr(strat, "_ema_latest", flat_ema)
        monkeypatch.setattr(strat, "_stoch_k", oversold_k)
        monkeypatch.setattr(strat, "_stoch_d", oversold_d)

    hyper_decision = hyper_strategy.evaluate(candles)
    assert hyper_decision.action == "LONG"
    assert hyper_strategy.latest_snapshot is not None
    assert hyper_strategy.latest_snapshot.get("trend_filter_active") is False

    aggressive_decision = aggressive_strategy.evaluate(candles)
    assert aggressive_decision.action == "FLAT"
    assert aggressive_strategy.latest_snapshot is not None
    assert aggressive_strategy.latest_snapshot.get("throttle_reason") == "trend_filter"


def test_hyper_aggressive_allows_fast_flips(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = EMAStochasticStrategy(
        ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3, params=ScalpingParams.hyper_aggressive())
    )

    def flat_ema(values, length):  # noqa: ANN001
        return 100.0

    def stoch_k(highs, lows, closes, idx):  # noqa: ANN001
        return 30.0 if closes[idx] < 103 else 80.0

    def stoch_d(highs, lows, closes, idx):  # noqa: ANN001
        return 40.0 if closes[idx] < 103 else 70.0

    monkeypatch.setattr(strategy, "_ema_latest", flat_ema)
    monkeypatch.setattr(strategy, "_stoch_k", stoch_k)
    monkeypatch.setattr(strategy, "_stoch_d", stoch_d)

    candles_long = _build_candles([100, 101, 102])
    candles_short = _build_candles([100, 101, 102, 103, 104])

    long_signal = strategy.evaluate(candles_long, position_side=Side.FLAT)
    assert long_signal.action == "LONG"

    short_signal = strategy.evaluate(candles_short, position_side=Side.FLAT)
    assert short_signal.action == "SHORT"


def test_scalping_strategy_triggers_short_on_downtrend_cross(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3, size_usd=750)
    strategy = EMAStochasticStrategy(config)
    candles = _build_candles([105, 104, 103, 102, 101])
    idx_prev = len(candles) - 2
    idx_now = len(candles) - 1

    def fake_stoch_k(highs, lows, closes, idx):  # noqa: ANN001
        return 80.0 if idx == idx_prev else 60.0

    def fake_stoch_d(highs, lows, closes, idx):  # noqa: ANN001
        return 70.0 if idx == idx_prev else 65.0

    def fake_ema(values, length):  # noqa: ANN001
        return 95.0 if length == strategy.config.fast_ema else 100.0

    monkeypatch.setattr(strategy, "_stoch_k", fake_stoch_k)
    monkeypatch.setattr(strategy, "_stoch_d", fake_stoch_d)
    monkeypatch.setattr(strategy, "_ema_latest", fake_ema)

    decision = strategy.evaluate(candles)

    assert decision.action == "SHORT"
    assert decision.reason == "short_trend_stoch_cross"
    assert decision.size_usd == pytest.approx(750)
    assert strategy.latest_snapshot is not None
    assert strategy.latest_snapshot.get("stoch_d_prev") == pytest.approx(70.0)


def test_scalping_strategy_needs_minimum_history() -> None:
    strategy = EMAStochasticStrategy(ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3))
    candles = _build_candles([100, 101])
    decision = strategy.evaluate(candles)
    assert decision.action == "FLAT"
    assert strategy.latest_snapshot is None


def test_scalping_strategy_closes_long_on_opposite_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3)
    strategy = EMAStochasticStrategy(config)
    candles = _build_candles([105, 104, 103, 102, 101])
    idx_prev = len(candles) - 2
    idx_now = len(candles) - 1

    def fake_stoch_k(highs, lows, closes, idx):  # noqa: ANN001
        return 85.0 if idx == idx_prev else 70.0

    def fake_stoch_d(highs, lows, closes, idx):  # noqa: ANN001
        return 80.0 if idx == idx_prev else 60.0

    def fake_ema(values, length):  # noqa: ANN001
        return 95.0 if length == strategy.config.fast_ema else 100.0

    monkeypatch.setattr(strategy, "_stoch_k", fake_stoch_k)
    monkeypatch.setattr(strategy, "_stoch_d", fake_stoch_d)
    monkeypatch.setattr(strategy, "_ema_latest", fake_ema)

    decision = strategy.evaluate(candles, position_side=Side.LONG)

    assert decision.action == "CLOSE_LONG"
    assert decision.reason == "close_long_opposite_signal"
    assert decision.size_usd == pytest.approx(0.0)


def test_scalping_strategy_holds_position_without_new_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3)
    strategy = EMAStochasticStrategy(config)
    candles = _build_candles([100, 101, 102, 103])
    idx_prev = len(candles) - 2
    idx_now = len(candles) - 1

    def fake_stoch_k(highs, lows, closes, idx):  # noqa: ANN001
        return 40.0 if idx == idx_now else 35.0

    def fake_stoch_d(highs, lows, closes, idx):  # noqa: ANN001
        return 50.0 if idx == idx_now else 55.0

    def fake_ema(values, length):  # noqa: ANN001
        return 105.0 if length == strategy.config.fast_ema else 100.0

    monkeypatch.setattr(strategy, "_stoch_k", fake_stoch_k)
    monkeypatch.setattr(strategy, "_stoch_d", fake_stoch_d)
    monkeypatch.setattr(strategy, "_ema_latest", fake_ema)

    decision = strategy.evaluate(candles, position_side=Side.LONG)

    assert decision.action == "HOLD"
    assert decision.reason == "hold_position_no_new_signal"
    assert decision.size_usd == pytest.approx(0.0)


def test_scalping_strategy_reenters_after_one_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3)
    strategy = EMAStochasticStrategy(config)
    candles = _build_candles([100, 101, 102, 103, 104, 105])

    def fake_stoch_k(highs, lows, closes, idx):  # noqa: ANN001
        current_idx = len(highs) - 1
        if idx == current_idx:
            return 30.0
        if idx == current_idx - 1:
            return 25.0
        return 50.0

    def fake_stoch_d(highs, lows, closes, idx):  # noqa: ANN001
        return 60.0

    def fake_ema(values, length):  # noqa: ANN001
        return 110.0 if length == config.fast_ema else 90.0

    monkeypatch.setattr(strategy, "_stoch_k", fake_stoch_k)
    monkeypatch.setattr(strategy, "_stoch_d", fake_stoch_d)
    monkeypatch.setattr(strategy, "_ema_latest", fake_ema)

    first_decision = strategy.evaluate(candles[:-1], position_side=Side.FLAT)
    assert first_decision.action == "LONG"

    second_decision = strategy.evaluate(candles, position_side=Side.FLAT)
    assert second_decision.action == "LONG"


def test_min_bars_between_trades_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    candles = _build_candles([100 + i for i in range(10)])

    def fake_stoch_k(highs, lows, closes, idx):  # noqa: ANN001
        return 30.0

    def fake_stoch_d(highs, lows, closes, idx):  # noqa: ANN001
        return 50.0

    def fake_ema(values, length):  # noqa: ANN001
        return 120.0 if length == 2 else 80.0

    aggressive_strategy = EMAStochasticStrategy(
        ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3, params=ScalpingParams.aggressive())
    )
    hyper_strategy = EMAStochasticStrategy(ScalpingConfig(fast_ema=2, slow_ema=3, k_period=3))

    for strat in (aggressive_strategy, hyper_strategy):
        monkeypatch.setattr(strat, "_stoch_k", fake_stoch_k)
        monkeypatch.setattr(strat, "_stoch_d", fake_stoch_d)
        monkeypatch.setattr(strat, "_ema_latest", fake_ema)

    first_aggressive = aggressive_strategy.evaluate(candles, position_side=Side.FLAT)
    assert first_aggressive.action == "LONG"
    second_aggressive = aggressive_strategy.evaluate(candles, position_side=Side.FLAT)
    assert second_aggressive.action == "FLAT"
    assert aggressive_strategy.latest_snapshot is not None
    assert aggressive_strategy.latest_snapshot.get("throttle_reason") == "cooldown"

    first_hyper = hyper_strategy.evaluate(candles, position_side=Side.FLAT)
    assert first_hyper.action == "LONG"
    second_hyper = hyper_strategy.evaluate(candles, position_side=Side.FLAT)
    assert second_hyper.action == "LONG"
    assert hyper_strategy.latest_snapshot is not None
    assert hyper_strategy.latest_snapshot.get("throttle_reason") is None
