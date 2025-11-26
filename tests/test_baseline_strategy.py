from datetime import datetime, timedelta

import pytest

from core.models import Candle
from strategies.baseline_rsi_trend import BaselineConfig, BaselineRSITrend


def _make_candles(closes):
    now = datetime.utcnow()
    candles = []
    for idx, close in enumerate(closes):
        open_time = now + timedelta(minutes=idx)
        candles.append(
            Candle(
                symbol="BTCUSDT",
                open_time=open_time,
                close_time=open_time + timedelta(minutes=1),
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1_000,
            )
        )
    return candles


def test_baseline_long_signal(monkeypatch):
    config = BaselineConfig(ma_length=3, rsi_length=2, size_usd=500)
    strategy = BaselineRSITrend(config)
    candles = _make_candles([100, 101, 105, 110])
    monkeypatch.setattr(BaselineRSITrend, "_compute_rsi", staticmethod(lambda *_: 20.0))

    decision = strategy.evaluate(candles)

    assert decision.action == "LONG"
    assert decision.size_usd == pytest.approx(500)
    assert decision.sl_pct == pytest.approx(config.stop_loss_pct)
    assert decision.tp_pct == pytest.approx(config.take_profit_pct)


def test_baseline_short_signal(monkeypatch):
    config = BaselineConfig(ma_length=3, rsi_length=2, size_usd=750)
    strategy = BaselineRSITrend(config)
    candles = _make_candles([120, 110, 105, 90])
    monkeypatch.setattr(BaselineRSITrend, "_compute_rsi", staticmethod(lambda *_: 80.0))

    decision = strategy.evaluate(candles)

    assert decision.action == "SHORT"
    assert decision.size_usd == pytest.approx(750)
    assert decision.sl_pct == pytest.approx(config.stop_loss_pct)
    assert decision.tp_pct == pytest.approx(config.take_profit_pct)


def test_baseline_flat_signal(monkeypatch):
    config = BaselineConfig(ma_length=3, rsi_length=2)
    strategy = BaselineRSITrend(config)
    candles = _make_candles([100, 100, 100, 100])
    monkeypatch.setattr(BaselineRSITrend, "_compute_rsi", staticmethod(lambda *_: 50.0))

    decision = strategy.evaluate(candles)

    assert decision.action == "FLAT"
    assert decision.size_usd == 0.0
    assert decision.sl_pct == pytest.approx(config.stop_loss_pct)
    assert decision.tp_pct == pytest.approx(config.take_profit_pct)
