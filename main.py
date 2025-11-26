import argparse
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from backtest.runner import BacktestRunner
from core.engine import TradingEngine
from core.llm_adapter import LLMAdapter
from core.models import Candle, RiskConfig
from core.risk import RiskManager
from core.safety import SafetyLimits
from core.state import PositionManager
from core.strategy import IndicatorConfig, Strategy
from exchange.binance_client import BinanceClient
from exchange.binance_stream import BinanceStream
from exchange.order_router import OrderRouter
from exchange.live_exchange import LiveExchange
from strategies.baseline_rsi_trend import BaselineConfig, BaselineRSITrend
from infra.config_loader import ConfigLoader
from infra.logging import bind_log_context, log_event, logger, setup_logging


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


def create_engine(config, run_mode: str):
    risk_config = RiskConfig(**config.get("risk", {}))
    safety_cfg = config.get("safety", {})
    safety_limits = SafetyLimits(
        max_daily_drawdown_pct=float(safety_cfg.get("max_daily_drawdown_pct", 5.0)),
        max_total_notional_usd=float(safety_cfg.get("max_total_notional_usd", 25_000.0)),
        max_consecutive_losses=int(safety_cfg.get("max_consecutive_losses", 3)),
    )
    risk_manager = RiskManager(risk_config, safety_limits=safety_limits)
    position_manager = PositionManager(run_mode=run_mode)
    llm_adapter = LLMAdapter()
    strategy_mode = config.get("strategy_mode", "llm").lower()
    baseline_strategy = None
    if strategy_mode not in {"llm", "baseline"}:
        strategy_mode = "llm"
    if strategy_mode == "baseline":
        baseline_cfg = config.get("baseline_strategy", {})
        defaults = BaselineConfig()
        baseline_config = BaselineConfig(
            ma_length=int(baseline_cfg.get("ma_length", defaults.ma_length)),
            rsi_length=int(baseline_cfg.get("rsi_length", defaults.rsi_length)),
            rsi_oversold=float(baseline_cfg.get("rsi_oversold", defaults.rsi_oversold)),
            rsi_overbought=float(baseline_cfg.get("rsi_overbought", defaults.rsi_overbought)),
            size_usd=float(baseline_cfg.get("size_usd", defaults.size_usd)),
            stop_loss_pct=float(baseline_cfg.get("stop_loss_pct", defaults.stop_loss_pct)),
            take_profit_pct=float(baseline_cfg.get("take_profit_pct", defaults.take_profit_pct)),
        )
        baseline_strategy = BaselineRSITrend(baseline_config)
        llm_adapter = None
    strategy = Strategy(
        IndicatorConfig(),
        llm_adapter=llm_adapter,
        strategy_mode=strategy_mode,
        baseline_strategy=baseline_strategy,
    )
    binance_cfg = config.get("binance", {})
    client = None
    order_router = None
    live_exchange = None
    stream = None
    if run_mode in {"paper", "live"}:
        client = BinanceClient(
            api_key=binance_cfg.get("api_key", ""),
            api_secret=binance_cfg.get("api_secret", ""),
            testnet=binance_cfg.get("testnet", True),
        )
        order_router = OrderRouter(client)
        live_exchange = LiveExchange(order_router, client, position_manager)
        stream = BinanceStream(symbol=config.get("symbol", "BTCUSDT"), interval="1m", testnet=binance_cfg.get("testnet", True))
    engine = TradingEngine(
        strategy,
        risk_manager,
        position_manager,
        order_router,
        stream=stream,
        safety_limits=safety_limits,
        exchange=live_exchange,
        run_mode=run_mode,
    )
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        help="Override RUN_MODE/config for this session",
    )
    args = parser.parse_args()

    setup_logging(log_to_file=True, state_file="logs/dashboard_state.json")
    config = ConfigLoader().load(mode_override=args.mode)
    run_mode = config.get("run_mode", "backtest")
    run_id = str(uuid4())
    bind_log_context(run_mode=run_mode, run_id=run_id)
    logger.info(
        "Starting bot",
        extra={"run_mode": run_mode, "testnet": config.get("binance", {}).get("testnet", True)},
    )
    log_event(
        "BOT_START",
        run_mode=run_mode,
        symbol=config.get("symbol", "BTCUSDT"),
        testnet=config.get("binance", {}).get("testnet", True),
        mode_override=args.mode,
    )

    if run_mode == "backtest":
        candles = load_sample_candles()
        engine = create_engine(config, run_mode)
        result = engine.run_backtest(candles)
        logger.info("Backtest run completed", extra={"metrics": result["metrics"]})
    else:
        engine = create_engine(config, run_mode)
        asyncio.run(engine.run_live())


if __name__ == "__main__":
    main()
