import argparse
import asyncio
from datetime import datetime, timedelta

from backtest.runner import BacktestRunner
from core.engine import TradingEngine
from core.llm_adapter import LLMAdapter
from core.models import Candle, RiskConfig
from core.risk import RiskManager
from core.state import PositionManager
from core.strategy import IndicatorConfig, Strategy
from exchange.binance_client import BinanceClient
from exchange.binance_stream import BinanceStream
from exchange.order_router import OrderRouter
from infra.config_loader import ConfigLoader
from infra.logging import logger


def load_sample_candles() -> list[Candle]:
    now = datetime.utcnow()
    candles = []
    price = 20_000
    for i in range(60):
        open_time = now - timedelta(minutes=60 - i)
        close_time = open_time + timedelta(minutes=1)
        price += 10
        candles.append(
            Candle(
                symbol="BTCUSDT",
                open_time=open_time,
                close_time=close_time,
                open=price - 5,
                high=price + 5,
                low=price - 10,
                close=price,
                volume=1_000,
            )
        )
    return candles


def create_engine(config):
    risk_config = RiskConfig()
    risk_manager = RiskManager(risk_config)
    position_manager = PositionManager()
    llm_adapter = LLMAdapter()
    strategy = Strategy(IndicatorConfig(), llm_adapter=llm_adapter)
    client = BinanceClient(
        api_key=config["binance"].get("api_key", ""),
        api_secret=config["binance"].get("api_secret", ""),
        testnet=config.get("testnet", True),
    )
    order_router = OrderRouter(client)
    stream = BinanceStream(symbol=config.get("symbol", "BTCUSDT"), interval="1m", testnet=config.get("testnet", True))
    engine = TradingEngine(strategy, risk_manager, position_manager, order_router, stream=stream)
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live trading")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    args = parser.parse_args()

    config = ConfigLoader().load()
    if args.live:
        if not config.get("live_trading_enabled", False):
            raise SystemExit("Live trading disabled: set live_trading_enabled true in config")
        engine = create_engine(config)
        asyncio.run(engine.run_live())
    else:
        candles = load_sample_candles()
        engine = create_engine(config)
        result = engine.run_backtest(candles)
        logger.info("Backtest run completed", extra={"metrics": result["metrics"]})


if __name__ == "__main__":
    main()
