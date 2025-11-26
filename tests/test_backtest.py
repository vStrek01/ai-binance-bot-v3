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


def test_backtest_equity_grows_on_trend():
    candles = generate_trending_candles()
    strategy = Strategy(IndicatorConfig(fast_ma=3, slow_ma=5), llm_adapter=None)
    risk = RiskManager(RiskConfig(max_risk_per_trade_pct=0.01))
    pm = PositionManager()
    runner = BacktestRunner(strategy, risk, pm, initial_equity=10000)
    result = runner.run(candles)
    assert result["metrics"]["final_equity"] > 10000
