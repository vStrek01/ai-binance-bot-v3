from datetime import datetime

from core.engine import TradingEngine
from core.models import RiskConfig, Signal, Side
from core.risk import RiskManager
from core.safety import SafetyLimits, SafetyState, cap_notional, check_kill_switch
from core.state import PositionManager
from infra import logging as logging_utils


class _DummyStrategy:
    strategy_mode = "dummy"

    def evaluate(self, market_state):  # pragma: no cover - not used in these tests
        return Signal(action=Side.FLAT)


def _engine_with_limits(
    limits: SafetyLimits,
    *,
    equity: float,
    losses: int = 0,
    start_equity: float = 10_000.0,
    run_mode: str = "backtest",
) -> TradingEngine:
    risk = RiskManager(RiskConfig(max_risk_per_trade_pct=0.01, max_daily_drawdown_pct=0.05))
    risk.daily_start_equity = start_equity
    risk.last_reset = datetime.utcnow()
    position_manager = PositionManager(equity=equity, consecutive_losses=losses)
    return TradingEngine(_DummyStrategy(), risk, position_manager, safety_limits=limits, run_mode=run_mode)


def _long_signal() -> Signal:
    return Signal(action=Side.LONG, confidence=1.0, stop_loss_pct=0.01, take_profit_pct=0.02)


def _limits() -> SafetyLimits:
    return SafetyLimits(max_daily_drawdown_pct=5.0, max_total_notional_usd=20_000.0, max_consecutive_losses=3)


def test_kill_switch_triggers_on_drawdown():
    limits = _limits()
    state = SafetyState(daily_start_equity=10_000.0, current_equity=9_400.0, consecutive_losses=0)
    assert check_kill_switch(limits, state)


def test_kill_switch_triggers_on_consecutive_losses():
    limits = _limits()
    state = SafetyState(daily_start_equity=10_000.0, current_equity=9_800.0, consecutive_losses=3)
    assert check_kill_switch(limits, state)


def test_notional_cap_respects_symbol_limit():
    qty = cap_notional(
        requested_qty=5,
        price=1_000.0,
        max_symbol_notional=4_000.0,
        max_total_notional=20_000.0,
        current_symbol_notional=3_500.0,
        current_total_notional=3_500.0,
    )
    assert round(qty * 1_000.0, 2) == 500.0


def test_notional_cap_respects_global_limit():
    qty = cap_notional(
        requested_qty=20,
        price=1_000.0,
        max_symbol_notional=50_000.0,
        max_total_notional=25_000.0,
        current_symbol_notional=0.0,
        current_total_notional=24_000.0,
    )
    assert round(qty * 1_000.0, 2) == 1_000.0


def test_kill_switch_blocks_orders_on_daily_drawdown():
    limits = SafetyLimits(max_daily_drawdown_pct=5.0, max_total_notional_usd=1_000_000.0, max_consecutive_losses=10)
    engine = _engine_with_limits(limits, equity=9_400.0)
    assert engine.map_signal_to_order(_long_signal(), price=100.0) is None
    assert engine.kill_switch_engaged


def test_kill_switch_blocks_orders_on_consecutive_losses():
    limits = SafetyLimits(max_daily_drawdown_pct=50.0, max_total_notional_usd=1_000_000.0, max_consecutive_losses=3)
    engine = _engine_with_limits(limits, equity=10_000.0, losses=3)
    assert engine.map_signal_to_order(_long_signal(), price=100.0) is None
    assert engine.kill_switch_engaged


def test_kill_switch_stops_future_orders_after_trigger():
    limits = SafetyLimits(max_daily_drawdown_pct=2.0, max_total_notional_usd=1_000_000.0, max_consecutive_losses=10)
    engine = _engine_with_limits(limits, equity=10_000.0)

    assert engine.map_signal_to_order(_long_signal(), price=100.0) is not None

    engine.position_manager.equity = 9_700.0
    assert engine.map_signal_to_order(_long_signal(), price=100.0) is None
    assert engine.kill_switch_engaged

    engine.position_manager.equity = 10_500.0
    assert engine.map_signal_to_order(_long_signal(), price=100.0) is None


def test_demo_live_kill_switch_logs_event(tmp_path):
    state_file = tmp_path / "obs_state.json"
    logging_utils._state_file = state_file
    logging_utils._recent_events.clear()
    limits = SafetyLimits(max_daily_drawdown_pct=5.0, max_total_notional_usd=1_000_000.0, max_consecutive_losses=5)
    engine = _engine_with_limits(limits, equity=9_400.0, run_mode="demo-live")
    assert engine.map_signal_to_order(_long_signal(), price=100.0) is None
    events = logging_utils.get_recent_events(limit=5)
    try:
        assert any(event.get("event") == "KILL_SWITCH_TRIGGERED" for event in events)
        assert any(event.get("run_mode") == "demo-live" for event in events)
    finally:
        logging_utils._state_file = None
        logging_utils._recent_events.clear()
