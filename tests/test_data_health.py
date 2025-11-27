from __future__ import annotations

import pytest

from typing import Any

from core.engine import TradingEngine
from core.models import RiskConfig, Signal, Side
from core.risk import RiskManager
from core.state import PositionManager
from core.strategy import IndicatorConfig, Strategy
from exchange.data_health import DataHealthMonitor, DataHealthStatus


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


class _StubStream:
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1m") -> None:
        self.symbol = symbol
        self.interval = interval
        self.history = []


class _StubHealth:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy
        self.calls = 0

    def is_data_stale(self, symbol: str, interval: str) -> DataHealthStatus:
        self.calls += 1
        return DataHealthStatus(
            healthy=self.healthy,
            seconds_since_update=0.0 if self.healthy else 999.0,
            threshold_seconds=60.0,
            last_update=0.0,
        )

    def is_healthy(self, symbol: str, interval: str) -> bool:
        return self.healthy

    def mark_update(self, symbol: str, interval: str, timestamp: object | None = None) -> None:  # pragma: no cover - shim
        return None


def _engine(data_health: Any, run_mode: str = "demo-live") -> TradingEngine:
    risk_manager = RiskManager(RiskConfig())
    position_manager = PositionManager(equity=10_000.0)
    stream = _StubStream()
    strategy = Strategy(IndicatorConfig())
    return TradingEngine(
        strategy,
        risk_manager,
        position_manager,
        stream=stream,
        run_mode=run_mode,
        data_health=data_health,
    )


def test_data_health_monitor_flags_stale_threshold():
    clock = _ManualClock()
    monitor = DataHealthMonitor(stale_multiplier=2.0, clock=clock)
    monitor.mark_update("BTCUSDT", "1m", timestamp=clock())
    status = monitor.is_data_stale("BTCUSDT", "1m")
    assert status.healthy
    clock.advance(130.0)
    status = monitor.is_data_stale("BTCUSDT", "1m")
    assert not status.healthy
    assert status.seconds_since_update == pytest.approx(130.0)
    assert status.threshold_seconds == pytest.approx(120.0)


def test_trading_engine_refuses_orders_when_data_unhealthy():
    monitor = _StubHealth(healthy=False)
    engine = _engine(monitor, run_mode="demo-live")
    signal = Signal(action=Side.LONG, stop_loss_pct=0.01, take_profit_pct=0.02)
    assert engine.map_signal_to_order(signal, price=100.0) is None


def test_backtest_mode_ignores_data_health_gate():
    monitor = _StubHealth(healthy=False)
    engine = _engine(monitor, run_mode="backtest")
    signal = Signal(action=Side.LONG, stop_loss_pct=0.01, take_profit_pct=0.02)
    order = engine.map_signal_to_order(signal, price=100.0)
    assert order is not None
    assert order.symbol == "BTCUSDT"