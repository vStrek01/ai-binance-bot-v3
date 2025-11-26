from __future__ import annotations

import asyncio
from typing import Callable, Iterable, Optional

from core.models import MarketState, OrderRequest, OrderType, Side
from core.risk import RiskManager
from core.state import PositionManager
from core.strategy import Strategy
from exchange.order_router import OrderRouter
from exchange.binance_stream import BinanceStream
from infra.logging import logger


class TradingEngine:
    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        order_router: OrderRouter,
        stream: Optional[BinanceStream] = None,
    ):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.order_router = order_router
        self.stream = stream

    def get_status(self):
        open_positions = self.position_manager.get_open_positions()
        equity = self.position_manager.equity
        drawdown = 0.0
        return {"equity": equity, "open_positions": [p.model_dump() for p in open_positions], "drawdown": drawdown}

    def map_signal_to_order(self, signal_action: Side, price: float) -> Optional[OrderRequest]:
        if signal_action == Side.FLAT:
            return None
        stop_loss = price * (0.99 if signal_action == Side.LONG else 1.01)
        take_profit = price * (1.02 if signal_action == Side.LONG else 0.98)
        side = signal_action
        return OrderRequest(
            symbol=self.stream.symbol if self.stream else "",
            side=side,
            order_type=OrderType.MARKET,
            quantity=0,  # risk manager fills
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=1,
        )

    async def run_live(self):
        if self.stream is None:
            raise RuntimeError("Binance stream not configured")
        async for candle in self.stream.candle_stream():
            if not candle:
                continue
            state = MarketState(
                symbol=candle.symbol,
                candles=self.stream.history,
                equity=self.position_manager.equity,
                open_positions=self.position_manager.get_open_positions(),
            )
            if candle.close_time != self.stream.history[-1].close_time:
                continue
            signal = self.strategy.evaluate(state)
            if signal.action == Side.FLAT:
                continue
            order_req = self.map_signal_to_order(signal.action, candle.close)
            if order_req is None:
                continue
            safe_order = self.risk_manager.validate(order_req, state)
            if safe_order:
                fill = await self.order_router.execute(safe_order)
                if fill:
                    self.position_manager.update_on_fill(
                        fill,
                        side=safe_order.side,
                        symbol=safe_order.symbol,
                        leverage=safe_order.leverage,
                        stop_loss=safe_order.stop_loss,
                        take_profit=safe_order.take_profit,
                    )

    def run_backtest(self, candles: Iterable) -> dict:
        from backtest.runner import BacktestRunner

        runner = BacktestRunner(self.strategy, self.risk_manager, self.position_manager)
        return runner.run(list(candles))
