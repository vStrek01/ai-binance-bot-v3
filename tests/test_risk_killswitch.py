from dataclasses import replace

from bot.core.config import load_config
from bot.risk import CloseTradeRequest, ExposureState, OpenTradeRequest, RiskEngine, TradeEvent, TradingMode


def _basic_open(symbol: str, equity: float) -> OpenTradeRequest:
    return OpenTradeRequest(
        symbol=symbol,
        side="BUY",
        quantity=0.001,
        price=100.0,
        available_balance=10_000.0,
        equity=equity,
        exposure=ExposureState(per_symbol={}, total=0.0),
        filters=None,
    )


def test_daily_loss_kill_switch_blocks_new_trades() -> None:
    cfg = load_config()
    cfg = replace(
        cfg,
        risk=replace(
            cfg.risk,
            max_daily_loss_pct=0.05,
            max_daily_loss_abs=None,
            stop_trading_on_daily_loss=True,
            close_positions_on_daily_loss=True,
        ),
    )
    engine = RiskEngine(cfg)
    engine.update_equity(1_000.0)
    engine.register_trade(TradeEvent(pnl=-60.0, equity=940.0, symbol="BTCUSDT"))

    decision = engine.evaluate_open(_basic_open("BTCUSDT", 940.0))
    state = engine.current_state()

    assert decision.allowed is False
    assert decision.should_flatten is True
    assert decision.reason == "daily_loss_pct"
    assert state.trading_mode == TradingMode.HALTED_DAILY_LOSS
    assert state.flatten_required is True


def test_flatten_decision_allowed_while_halted() -> None:
    cfg = load_config()
    cfg = replace(
        cfg,
        risk=replace(
            cfg.risk,
            max_daily_loss_pct=0.01,
            close_positions_on_daily_loss=True,
            stop_trading_on_daily_loss=True,
        ),
    )
    engine = RiskEngine(cfg)
    engine.update_equity(1_000.0)
    engine.register_trade(TradeEvent(pnl=-15.0, equity=985.0, symbol="ETHUSDT"))

    close_decision = engine.evaluate_close(
        CloseTradeRequest(
            symbol="ETHUSDT",
            quantity=0.01,
            price=100.0,
            equity=985.0,
            exposure=ExposureState(per_symbol={}, total=0.0),
            filters=None,
        )
    )

    assert close_decision.allowed is True
    assert close_decision.should_flatten is True


def test_reset_daily_limits_restores_trading() -> None:
    cfg = load_config()
    cfg = replace(
        cfg,
        risk=replace(
            cfg.risk,
            max_daily_loss_pct=0.02,
            stop_trading_on_daily_loss=True,
            close_positions_on_daily_loss=True,
        ),
    )
    engine = RiskEngine(cfg)
    engine.update_equity(1_000.0)
    engine.register_trade(TradeEvent(pnl=-25.0, equity=975.0, symbol="LTCUSDT"))

    assert engine.current_state().trading_mode == TradingMode.HALTED_DAILY_LOSS

    engine.reset_daily_limits()
    open_decision = engine.evaluate_open(_basic_open("LTCUSDT", 975.0))

    assert engine.current_state().trading_mode == TradingMode.NORMAL
    assert open_decision.allowed is True
    assert open_decision.reason is None


def test_safety_clamps_apply_for_live_modes() -> None:
    cfg = load_config()
    cfg = replace(
        cfg,
        risk=replace(cfg.risk, leverage=100.0, max_daily_loss_pct=0.8),
        runtime=replace(cfg.runtime, dry_run=False, live_trading=True),
    )
    engine = RiskEngine(cfg)

    assert engine._config.risk.leverage <= 25.0
    assert engine._config.risk.max_daily_loss_pct <= 0.2
