from datetime import datetime, timedelta

from backtest.runner import BacktestRunner
from core.models import Candle, RiskConfig
from core.risk import RiskManager
from core.safety import SafetyLimits
from core.state import PositionManager
from core.strategy import IndicatorConfig, Strategy
from strategies.baseline_rsi_trend import BaselineConfig, BaselineRSITrend


def _build_trend_candles(count: int = 20):
    now = datetime.utcnow()
    candles = []
    price = 100.0
    for idx in range(count):
        price += 2.0
        open_time = now + timedelta(minutes=idx)
        candles.append(
            Candle(
                symbol="BTCUSDT",
                open_time=open_time,
                close_time=open_time + timedelta(minutes=1),
                open=price - 1,
                high=price + 1,
                low=price - 2,
                close=price,
                volume=1_000,
            )
        )
    return candles


def test_baseline_backtest_positive_pnl(monkeypatch):
    candles = _build_trend_candles()
    baseline_cfg = BaselineConfig(ma_length=3, rsi_length=2, size_usd=500)
    baseline = BaselineRSITrend(baseline_cfg)
    monkeypatch.setattr(BaselineRSITrend, "_compute_rsi", staticmethod(lambda *_: 20.0))

    strategy = Strategy(IndicatorConfig(), strategy_mode="baseline", baseline_strategy=baseline)
    risk_config = RiskConfig(max_risk_per_trade_pct=0.02, max_daily_drawdown_pct=0.5)
    safety_limits = SafetyLimits(max_daily_drawdown_pct=50.0, max_total_notional_usd=1_000_000.0, max_consecutive_losses=10)
    risk_manager = RiskManager(risk_config, safety_limits=safety_limits)
    position_manager = PositionManager(equity=10_000.0)

    runner = BacktestRunner(strategy, risk_manager, position_manager, initial_equity=10_000.0, safety_limits=safety_limits)
    result = runner.run(candles)

    assert any(p != 0 for p in result["trades"])
    assert sum(result["trades"]) > 0
    assert not runner.kill_switch_engaged
