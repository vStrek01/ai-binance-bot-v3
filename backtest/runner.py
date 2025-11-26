from __future__ import annotations

from typing import List

from backtest.execution_sim import ExecutionSimulator
from backtest.metrics import average_r, exposure, max_drawdown, sharpe_ratio, win_rate
from core.models import Candle, MarketState, OrderRequest, OrderType, Side
from core.risk import RiskManager
from core.state import PositionManager
from core.strategy import Strategy
from infra.logging import logger
from infra.persistence import save_backtest_results


class BacktestRunner:
    def __init__(self, strategy: Strategy, risk_manager: RiskManager, position_manager: PositionManager, initial_equity: float = 10_000):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.position_manager.equity = initial_equity
        self.simulator = ExecutionSimulator(spread=0.5, slippage=risk_manager.config.slippage, fee_rate=risk_manager.config.taker_fee_rate)

    def _map_signal_to_order(self, signal_action: Side, candle: Candle) -> OrderRequest:
        stop_loss = candle.close * (0.99 if signal_action == Side.LONG else 1.01)
        take_profit = candle.close * (1.02 if signal_action == Side.LONG else 0.98)
        return OrderRequest(
            symbol=candle.symbol,
            side=signal_action,
            order_type=OrderType.MARKET,
            quantity=0,
            price=candle.close,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def run(self, candles: List[Candle]):
        equity_curve = []
        trade_pnls = []
        risks = []
        open_bars = []

        for idx, candle in enumerate(candles):
            market_state = MarketState(
                symbol=candle.symbol,
                candles=candles[: idx + 1],
                equity=self.position_manager.equity,
                open_positions=self.position_manager.get_open_positions(),
            )
            if candle.close_time < candles[idx].close_time and idx > 0:
                continue

            signal = self.strategy.evaluate(market_state)
            if signal.action != Side.FLAT:
                order = self._map_signal_to_order(signal.action, candle)
                safe_order = self.risk_manager.validate(order, market_state)
                if safe_order:
                    fill, fee = self.simulator.simulate(safe_order, candle.close)
                    self.position_manager.apply_fee(fee)
                    self.position_manager.update_on_fill(
                        fill,
                        side=safe_order.side,
                        symbol=safe_order.symbol,
                        leverage=safe_order.leverage,
                        stop_loss=safe_order.stop_loss,
                        take_profit=safe_order.take_profit,
                    )
                    trade_pnls.append(0.0)
                    risks.append(self.position_manager.equity * self.risk_manager.config.max_risk_per_trade_pct)

            # simple exit rule: close if opposite signal or at final candle
            open_positions = self.position_manager.get_open_positions()
            for pos in open_positions:
                should_close = False
                if signal.action == Side.FLAT:
                    should_close = True
                elif signal.action != pos.side:
                    should_close = True
                if idx == len(candles) - 1:
                    should_close = True
                if should_close:
                    exit_price = candle.close
                    pnl = (exit_price - pos.entry_price) * pos.quantity
                    if pos.side == Side.SHORT:
                        pnl = -pnl
                    self.position_manager.close_position(pos.symbol, exit_price)
                    trade_pnls[-1] = pnl

            equity_curve.append(self.position_manager.equity)
            open_bars.append(len(self.position_manager.get_open_positions()))

        metrics = {
            "final_equity": self.position_manager.equity,
            "max_drawdown": max_drawdown(equity_curve),
            "sharpe": sharpe_ratio([equity_curve[i] - equity_curve[i - 1] for i in range(1, len(equity_curve))]),
            "win_rate": win_rate(trade_pnls),
            "avg_r": average_r(trade_pnls, risks),
            "exposure": exposure(open_bars, len(open_bars)),
        }
        save_backtest_results({"equity_curve": equity_curve, "metrics": metrics, "trades": trade_pnls})
        logger.info("Backtest complete", extra={"metrics": metrics})
        return {"equity_curve": equity_curve, "metrics": metrics, "trades": trade_pnls}
