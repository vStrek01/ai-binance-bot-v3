from __future__ import annotations

from bot.risk.engine import RiskEngine, TradeEvent
from bot.risk.spec import build_core_risk_config
from core.risk import RiskManager
from infra.config_loader import load_config


def _calibrated_config(tmp_path, **risk_overrides):
    cfg = load_config(base_dir=tmp_path, mode_override="demo-live")
    cfg = cfg.model_copy(update={"risk": cfg.risk.model_copy(update=risk_overrides)})
    return cfg


def test_risk_parity_drawdown(tmp_path):
    cfg = _calibrated_config(
        tmp_path,
        max_daily_loss_pct=0.01,
        per_trade_risk=0.01,
        max_consecutive_losses=10,
        max_trades_per_day=20,
    )
    risk_manager = RiskManager(build_core_risk_config(cfg))
    engine = RiskEngine(cfg)
    starting_equity = 10_000.0
    risk_manager.reset_day_if_needed(starting_equity)
    engine.update_equity(starting_equity)

    equity = starting_equity
    pnl_sequence = (-150.0, -50.0, -25.0)
    triggered = None
    for idx, pnl in enumerate(pnl_sequence, start=1):
        risk_manager.record_trade_entry()
        engine.record_trade_entry()
        equity += pnl
        risk_manager.record_fill(pnl)
        engine.register_trade(TradeEvent(pnl=pnl, equity=equity))
        assert risk_manager.kill_switch_active == engine.trading_paused
        if risk_manager.kill_switch_active:
            triggered = idx
            break
    assert triggered is not None, "expected daily drawdown to trigger kill switch"
    assert engine.halt_reason is not None
    assert "daily_drawdown" in risk_manager.kill_switch_reason
    assert "daily_drawdown" in engine.halt_reason


def test_risk_parity_trade_limit(tmp_path):
    cfg = _calibrated_config(
        tmp_path,
        max_trades_per_day=2,
        per_trade_risk=0.005,
        max_daily_loss_pct=0.5,
        max_consecutive_losses=10,
    )
    risk_manager = RiskManager(build_core_risk_config(cfg))
    engine = RiskEngine(cfg)
    starting_equity = 5_000.0
    risk_manager.reset_day_if_needed(starting_equity)
    engine.update_equity(starting_equity)

    risk_manager.record_trade_entry()
    engine.record_trade_entry()
    assert not risk_manager.kill_switch_active
    assert not engine.trading_paused

    risk_manager.record_trade_entry()
    engine.record_trade_entry()

    assert risk_manager.kill_switch_active
    assert engine.trading_paused
    assert "trade_limit" in (risk_manager.kill_switch_reason or "")
    assert "trade_limit" in (engine.halt_reason or "")
