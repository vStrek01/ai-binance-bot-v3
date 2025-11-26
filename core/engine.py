from __future__ import annotations

import asyncio
from typing import Callable, Iterable, Optional

from core.models import MarketState, OrderRequest, OrderType, Side, Signal
from core.risk import RiskManager
from core.state import PositionManager
from core.strategy import Strategy
from core.safety import SafetyLimits, SafetyState, check_kill_switch
from exchange.order_router import OrderRouter
from exchange.binance_stream import BinanceStream
from exchange.base import Exchange
from infra.logging import logger, log_event


class TradingEngine:
    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        order_router: Optional[OrderRouter] = None,
        stream: Optional[BinanceStream] = None,
        safety_limits: Optional[SafetyLimits] = None,
        exchange: Optional[Exchange] = None,
        run_mode: str = "backtest",
    ):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.order_router = order_router
        self.stream = stream
        self.safety_limits = safety_limits
        self.kill_switch_engaged = False
        self.exchange = exchange
        self.run_mode = run_mode

    def get_status(self):
        open_positions = self.position_manager.get_open_positions()
        equity = self.position_manager.equity
        drawdown = 0.0
        return {"equity": equity, "open_positions": [p.model_dump() for p in open_positions], "drawdown": drawdown}

    def map_signal_to_order(self, signal: Signal, price: float) -> Optional[OrderRequest]:
        if self._kill_switch_active():
            return None
        if signal.action == Side.FLAT:
            return None
        sl_pct = signal.stop_loss_pct or 0.01
        tp_pct = signal.take_profit_pct or 0.02
        if signal.action == Side.LONG:
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
        else:
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)
        side = signal.action
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
        if self.order_router is None:
            raise RuntimeError("Order router not configured for live trading")
        async for candle in self.stream.candle_stream():
            if not candle:
                continue
            state = MarketState(
                symbol=candle.symbol,
                candles=self.stream.history,
                equity=self.position_manager.equity,
                open_positions=self.position_manager.get_open_positions(),
            )
            self._log_equity_snapshot(symbol=candle.symbol, mark_price=candle.close)
            if candle.close_time != self.stream.history[-1].close_time:
                continue
            self.risk_manager.reset_day_if_needed(self.position_manager.equity)
            if self._kill_switch_active():
                continue
            signal = self.strategy.evaluate(state)
            if signal.action == Side.FLAT:
                continue
            order_req = self.map_signal_to_order(signal, candle.close)
            if order_req is None:
                continue
            safe_order = self.risk_manager.validate(order_req, state)
            if safe_order:
                fill = await self._execute_order(safe_order)
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

        if hasattr(self.risk_manager, "enable_backtest_mode"):
            self.risk_manager.enable_backtest_mode()
        runner = BacktestRunner(
            self.strategy,
            self.risk_manager,
            self.position_manager,
            safety_limits=self.safety_limits,
            run_mode=self.run_mode,
        )
        return runner.run(list(candles))

    def _kill_switch_active(self) -> bool:
        if self.safety_limits is None:
            return False
        safety_state = self._build_safety_state()
        triggered, reason = check_kill_switch(self.safety_limits, safety_state, return_reason=True)
        if triggered and not self.kill_switch_engaged:
            self.kill_switch_engaged = True
            logger.error(
                "KILL_SWITCH_TRIGGERED",
                extra={
                    "equity": safety_state.current_equity,
                    "daily_start_equity": safety_state.daily_start_equity,
                    "consecutive_losses": safety_state.consecutive_losses,
                },
            )
            log_event(
                "KILL_SWITCH_TRIGGERED",
                reason=reason,
                equity=safety_state.current_equity,
                daily_start_equity=safety_state.daily_start_equity,
                consecutive_losses=safety_state.consecutive_losses,
                run_mode=self.run_mode,
            )
        return triggered

    def _build_safety_state(self) -> SafetyState:
        self.risk_manager.reset_day_if_needed(self.position_manager.equity)
        if self.risk_manager.daily_start_equity is None:
            self.risk_manager.daily_start_equity = self.position_manager.equity
        return SafetyState(
            daily_start_equity=self.risk_manager.daily_start_equity,
            current_equity=self.position_manager.equity,
            consecutive_losses=self.position_manager.consecutive_losses,
        )

    async def _execute_order(self, order: OrderRequest):
        if self.exchange is not None:
            return await self.exchange.place_order(order)
        if self.order_router is None:
            raise RuntimeError("No exchange or order router configured for execution")
        return await self.order_router.execute(order)

    def _log_equity_snapshot(self, *, symbol: Optional[str] = None, mark_price: Optional[float] = None) -> None:
        log_event(
            "EQUITY_SNAPSHOT",
            symbol=symbol,
            mark_price=mark_price,
            equity=self.position_manager.equity,
            balance=self.position_manager.equity,
            unrealized_pnl=0.0,
            open_positions=len(self.position_manager.get_open_positions()),
            run_mode=self.run_mode,
        )
