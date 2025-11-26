from bot.risk.engine import ExposureState, OpenTradeRequest, RiskEngine, TradeEvent, TradingMode
from core.models import MarketState, OrderRequest, OrderType, RiskConfig, Side
from core.risk import RiskManager
from infra.config_loader import load_config


def test_position_sizing_and_limits():
    config = RiskConfig(max_risk_per_trade_pct=0.01, max_daily_drawdown_pct=0.05, max_open_positions=1, max_leverage=5)
    risk = RiskManager(config)
    state = MarketState(symbol="BTCUSDT", candles=[], equity=10000, open_positions=[])
    order = OrderRequest(symbol="BTCUSDT", side=Side.LONG, order_type=OrderType.MARKET, quantity=0, price=100, stop_loss=99)
    safe_order = risk.validate(order, state)
    assert safe_order is not None
    assert round(safe_order.quantity, 2) == 100.0


def test_daily_drawdown_block():
    config = RiskConfig(max_daily_drawdown_pct=0.02)
    risk = RiskManager(config)
    risk.daily_start_equity = 10000
    state = MarketState(symbol="BTCUSDT", candles=[], equity=9700, open_positions=[])
    order = OrderRequest(symbol="BTCUSDT", side=Side.LONG, order_type=OrderType.MARKET, quantity=0, price=100, stop_loss=99)
    assert risk.validate(order, state) is None


def test_demo_live_oversized_position_rejected(tmp_path):
    cfg = load_config(base_dir=tmp_path, mode_override="demo-live")
    engine = RiskEngine(cfg)
    exposure = ExposureState(per_symbol={}, total=0.0)
    request = OpenTradeRequest(
        symbol="BTCUSDT",
        side="BUY",
        quantity=10.0,
        price=100.0,
        available_balance=1_000.0,
        equity=1_000.0,
        exposure=exposure,
        filters=None,
    )
    decision = engine.evaluate_open(request)
    assert not decision.allowed
    assert decision.reason == "symbol_abs_cap"


def test_demo_live_halts_on_daily_loss_and_loss_streak(tmp_path):
    cfg = load_config(base_dir=tmp_path, mode_override="demo-live")
    engine = RiskEngine(cfg)
    exposure = ExposureState(per_symbol={}, total=0.0)
    engine.update_equity(1_000.0)
    engine.register_trade(TradeEvent(pnl=-120.0, equity=880.0))
    blocked = engine.evaluate_open(
        OpenTradeRequest(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=100.0,
            available_balance=1_000.0,
            equity=880.0,
            exposure=exposure,
            filters=None,
        )
    )
    assert not blocked.allowed
    assert blocked.reason == "daily_loss_pct"
    assert engine.current_state().trading_mode == TradingMode.HALTED_DAILY_LOSS

    relaxed_cfg = cfg.model_copy(update={"risk": cfg.risk.model_copy(update={"max_daily_loss_pct": 0.5})})
    streak_engine = RiskEngine(relaxed_cfg)
    streak_engine.update_equity(1_000.0)
    equities = (995.0, 990.0, 985.0)
    for idx, equity in enumerate(equities, start=1):
        streak_engine.register_trade(TradeEvent(pnl=-5.0, equity=equity, symbol=f"SYM{idx}"))
    streak_block = streak_engine.evaluate_open(
        OpenTradeRequest(
            symbol="ETHUSDT",
            side="BUY",
            quantity=1.0,
            price=100.0,
            available_balance=1_000.0,
            equity=985.0,
            exposure=exposure,
            filters=None,
        )
    )
    assert not streak_block.allowed
    assert streak_block.reason == "loss_streak"
    assert streak_engine.current_state().trading_mode == TradingMode.HALTED_LOSS_STREAK
