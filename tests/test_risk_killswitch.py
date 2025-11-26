from bot.core import config
from bot.risk import RiskEngine


def test_daily_loss_kill_switch_blocks_new_trades() -> None:
    config.risk.max_daily_loss_pct = 0.05
    config.risk.max_daily_loss_abs = None
    config.risk.stop_trading_on_daily_loss = True
    config.risk.close_positions_on_daily_loss = True
    engine = RiskEngine()
    engine.update_equity(1_000)

    engine.register_trade(-60.0, equity=940.0)

    allowed, reason = engine.can_open_new_trades()
    snapshot = engine.snapshot()

    assert allowed is False
    assert reason == "daily_loss_pct"
    assert snapshot["trading_paused"] is True
    assert snapshot["reason"] == "daily_loss_pct"
    assert engine.should_flatten_positions() is True

    engine.reset_daily_limits()

    allowed_after_reset, _ = engine.can_open_new_trades()
    assert allowed_after_reset is True
    assert engine.should_flatten_positions() is False
