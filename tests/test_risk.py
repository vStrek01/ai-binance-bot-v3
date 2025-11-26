from core.models import MarketState, OrderRequest, OrderType, RiskConfig, Side
from core.risk import RiskManager


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
