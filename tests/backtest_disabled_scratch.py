from datetime import datetime, timedelta

from backtest.runner import BacktestRunner
from core.models import Candle, RiskConfig
from core.risk import RiskManager
from core.state import PositionManager
from core.strategy import IndicatorConfig, Strategy


def generate_trending_candles(count: int = 50):
    now = datetime.utcnow()
    candles = []
    price = 100
    for i in range(count):
        if i and i % 6 == 0:
            price -= 1
        open_time = now + timedelta(minutes=i)
        close_time = open_time + timedelta(minutes=1)
        price += 1
        candles.append(
            Candle(
                symbol="BTCUSDT",
                open_time=open_time,
                close_time=close_time,
                open=price,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=10,
            )
        )
    return candles


def generate_sideways_candles(count: int = 60):
    now = datetime.utcnow()
    candles = []
    price = 100
    for i in range(count):
        drift = 0.2 if i % 2 == 0 else -0.2
        price += drift
        open_time = now + timedelta(minutes=i)
        close_time = open_time + timedelta(minutes=1)
        candles.append(
            Candle(
                symbol="BTCUSDT",
                open_time=open_time,
                close_time=close_time,
                open=price,
                high=price + 0.25,
                low=price - 0.25,
                close=price + 0.05,
                volume=8,
            )
        )
    return candles


def test_backtest_equity_grows_on_trend():
    candles = generate_trending_candles()
    cfg = IndicatorConfig(
        fast_ema=3,
        slow_ema=8,
        rsi_length=5,
        rsi_overbought=80,
        rsi_oversold=20,
        atr_period=5,
        atr_stop=0.8,
        atr_target=1.5,
        cooldown_bars=1,
        hold_bars=20,
        pullback_atr_multiplier=0.0,
        min_confidence=0.2,
    )
    strategy = Strategy(cfg, llm_adapter=None)
    risk = RiskManager(RiskConfig(max_risk_per_trade_pct=0.01))
    pm = PositionManager()
    runner = BacktestRunner(strategy, risk, pm, initial_equity=10000)
    result = runner.run(candles)
    assert result["metrics"]["final_equity"] > 10000


def test_strategy_stays_flat_in_sideways_range():
    candles = generate_sideways_candles()
    strategy = Strategy(IndicatorConfig(), llm_adapter=None)
    risk = RiskManager(RiskConfig(max_risk_per_trade_pct=0.01))
    pm = PositionManager()
    runner = BacktestRunner(strategy, risk, pm, initial_equity=10000)
    result = runner.run(candles)
    assert result["trades"] == []
    assert result["metrics"]["final_equity"] == 10000
