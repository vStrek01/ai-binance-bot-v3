from __future__ import annotations

import json
import asyncio
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from backtest.metrics import average_r, average_win_loss, exposure, max_drawdown, sharpe_ratio, win_rate
from core.models import Candle, MarketState, OrderRequest, OrderType, Position, Side, Signal
from core.risk import RiskManager
from core.state import PositionManager
from core.strategy import Strategy
from core.safety import SafetyLimits, SafetyState, check_kill_switch
from exchange.simulated_exchange import SimulatedExchange
from infra.logging import logger, log_event


@dataclass(slots=True)
class BacktestSummary:
    starting_equity: float
    ending_equity: float
    net_pnl: float
    net_pnl_pct: float
    closed_trades: int
    win_rate_pct: float
    max_drawdown_pct: float
    avg_r_multiple: float
    avg_win: float
    avg_loss: float
    exposure_pct: float
    sharpe_ratio: float

    def as_dict(self) -> dict:
        return asdict(self)


class BacktestRunner:
    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        initial_equity: float = 10_000,
        safety_limits: SafetyLimits | None = None,
        run_mode: str = "backtest",
    ):
        self.strategy = strategy
        self.risk_manager = risk_manager
        if hasattr(self.risk_manager, "enable_backtest_mode"):
            self.risk_manager.enable_backtest_mode()
        self.position_manager = position_manager
        self.initial_equity = initial_equity
        self.position_manager.equity = initial_equity
        self.exchange = SimulatedExchange(
            self.position_manager,
            spread=0.5,
            slippage=risk_manager.config.slippage,
        )
        self.safety_limits = safety_limits
        self.kill_switch_engaged = False
        self.active_trades: Dict[str, int] = {}
        self.run_mode = run_mode

    def _map_signal_to_order(self, signal: Signal, candle: Candle) -> OrderRequest:
        sl_pct = signal.stop_loss_pct or 0.01
        tp_pct = signal.take_profit_pct or 0.02
        if signal.action == Side.LONG:
            stop_loss = candle.close * (1 - sl_pct)
            take_profit = candle.close * (1 + tp_pct)
        else:
            stop_loss = candle.close * (1 + sl_pct)
            take_profit = candle.close * (1 - tp_pct)
        return OrderRequest(
            symbol=candle.symbol,
            side=signal.action,
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
            self.exchange.update_market(candle)
            self._log_equity_snapshot(candle)
            if idx > 0 and candle.close_time <= candles[idx - 1].close_time:
                logger.debug(
                    "Skipping stale candle",
                    extra={"idx": idx, "close_time": candle.close_time, "prev_close": candles[idx - 1].close_time},
                )
                continue

            self.risk_manager.reset_day_if_needed(self.position_manager.equity)
            self._apply_protective_exits(candle, trade_pnls)
            market_state = MarketState(
                symbol=candle.symbol,
                candles=candles[: idx + 1],
                equity=self.position_manager.equity,
                open_positions=self.position_manager.get_open_positions(),
            )
            if self._kill_switch_active():
                equity_curve.append(self.position_manager.equity)
                open_bars.append(len(self.position_manager.get_open_positions()))
                continue

            signal = self.strategy.evaluate(market_state)
            if signal.action != Side.FLAT:
                logger.debug(
                    "Strategy signal",
                    extra={
                        "idx": idx,
                        "action": signal.action.value,
                        "price": candle.close,
                        "confidence": signal.confidence,
                    },
                )
                order = self._map_signal_to_order(signal, candle)
                safe_order = self.risk_manager.validate(order, market_state)
                if safe_order:
                    logger.debug(
                        "Order approved by risk",
                        extra={
                            "symbol": safe_order.symbol,
                            "side": safe_order.side.value,
                            "qty": safe_order.quantity,
                            "price": safe_order.price,
                        },
                    )
                    before_open = any(
                        pos.symbol == safe_order.symbol and pos.is_open()
                        for pos in self.position_manager.get_open_positions()
                    )
                    fill = asyncio.run(self.exchange.place_order(safe_order))
                    if fill:
                        entry_fee = abs(fill.avg_price * fill.filled_qty) * self.risk_manager.config.taker_fee_rate
                        self.position_manager.apply_fee(entry_fee)
                        if not before_open:
                            trade_idx = len(trade_pnls)
                            trade_pnls.append(0.0)
                            risks.append(self.position_manager.equity * self.risk_manager.config.max_risk_per_trade_pct)
                            self.active_trades[safe_order.symbol] = trade_idx
                else:
                    logger.debug(
                        "Order blocked by risk",
                        extra={"symbol": order.symbol, "side": order.side.value, "price": order.price},
                    )

            self._maybe_close_on_signal(signal.action, candle, idx, len(candles), trade_pnls)

            equity_curve.append(self.position_manager.equity)
            open_bars.append(len(self.position_manager.get_open_positions()))

        equity_returns = [equity_curve[i] - equity_curve[i - 1] for i in range(1, len(equity_curve))]
        sharpe = sharpe_ratio(equity_returns)
        avg_win, avg_loss = average_win_loss(trade_pnls)
        time_in_market = exposure(open_bars, len(open_bars))
        net_pnl = self.position_manager.equity - self.initial_equity
        metrics = {
            "final_equity": self.position_manager.equity,
            "total_pnl": net_pnl,
            "max_drawdown": max_drawdown(equity_curve),
            "sharpe": sharpe,
            "win_rate": win_rate(trade_pnls),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_r": average_r(trade_pnls, risks),
            "exposure": time_in_market,
            "trades": len(trade_pnls),
        }
        summary = self._build_summary(
            equity_curve,
            trade_pnls,
            risks,
            avg_win=avg_win,
            avg_loss=avg_loss,
            exposure_pct=time_in_market * 100,
            sharpe=sharpe,
        ).as_dict()
        logger.info("Backtest metrics: %s", json.dumps(summary))
        logger.info("Backtest complete", extra={"metrics": metrics})
        if candles:
            log_event(
                "BACKTEST_SUMMARY",
                symbol=candles[-1].symbol,
                start=candles[0].open_time,
                end=candles[-1].close_time,
                pnl=net_pnl,
                max_drawdown=metrics["max_drawdown"],
                trades=metrics["trades"],
                win_rate=metrics["win_rate"],
                sharpe=metrics["sharpe"],
                strategy_mode=self.strategy.strategy_mode if hasattr(self.strategy, "strategy_mode") else "unknown",
                run_mode=self.run_mode,
                summary=summary,
            )
        return {"equity_curve": equity_curve, "metrics": metrics, "trades": trade_pnls, "summary": summary}

    def _kill_switch_active(self) -> bool:
        if self.safety_limits is None:
            return False
        state = SafetyState(
            daily_start_equity=self.risk_manager.daily_start_equity or self.position_manager.equity,
            current_equity=self.position_manager.equity,
            consecutive_losses=self.position_manager.consecutive_losses,
        )
        triggered, reason = check_kill_switch(self.safety_limits, state, return_reason=True)
        if triggered and not self.kill_switch_engaged:
            self.kill_switch_engaged = True
            logger.error(
                "KILL_SWITCH_TRIGGERED",
                extra={
                    "equity": state.current_equity,
                    "daily_start_equity": state.daily_start_equity,
                    "consecutive_losses": state.consecutive_losses,
                },
            )
            log_event(
                "KILL_SWITCH_TRIGGERED",
                reason=reason,
                equity=state.current_equity,
                daily_start_equity=state.daily_start_equity,
                consecutive_losses=state.consecutive_losses,
                run_mode=self.run_mode,
            )
        return triggered

    def _apply_protective_exits(self, candle: Candle, trade_pnls: List[float]) -> None:
        for pos in list(self.position_manager.get_open_positions()):
            exit_price, reason = self._protective_hit(pos, candle)
            if exit_price is None:
                continue
            realized = self.position_manager.close_position(pos.symbol, exit_price)
            if realized is None:
                continue
            exit_fee = abs(exit_price * pos.quantity) * self.risk_manager.config.taker_fee_rate
            self.position_manager.apply_fee(exit_fee)
            self._record_trade_pnl(pos.symbol, realized, trade_pnls)
            logger.debug(
                "Closed position",
                extra={
                    "symbol": pos.symbol,
                    "side": pos.side.value,
                    "exit_price": exit_price,
                    "pnl": realized,
                    "reason": reason,
                },
            )

    def _protective_hit(self, pos: Position, candle: Candle) -> tuple[Optional[float], Optional[str]]:
        stop_hit = False
        tp_hit = False
        if pos.stop_loss is not None:
            if pos.side == Side.LONG:
                stop_hit = candle.low <= pos.stop_loss <= candle.high
            else:
                stop_hit = candle.high >= pos.stop_loss >= candle.low
        if pos.take_profit is not None:
            if pos.side == Side.LONG:
                tp_hit = candle.low <= pos.take_profit <= candle.high
            else:
                tp_hit = candle.low <= pos.take_profit <= candle.high
        if stop_hit:
            return pos.stop_loss, "stop_loss"
        if tp_hit:
            return pos.take_profit, "take_profit"
        return None, None

    def _maybe_close_on_signal(self, signal: Side, candle: Candle, idx: int, total: int, trade_pnls: List[float]) -> None:
        for pos in list(self.position_manager.get_open_positions()):
            should_close = False
            if signal == Side.FLAT:
                should_close = True
            elif signal != pos.side:
                should_close = True
            if idx == total - 1:
                should_close = True
            if not should_close:
                continue
            exit_price = candle.close
            realized = self.position_manager.close_position(pos.symbol, exit_price)
            if realized is None:
                continue
            exit_fee = abs(exit_price * pos.quantity) * self.risk_manager.config.taker_fee_rate
            self.position_manager.apply_fee(exit_fee)
            self._record_trade_pnl(pos.symbol, realized, trade_pnls)
            logger.debug(
                "Closed position",
                extra={
                    "symbol": pos.symbol,
                    "side": pos.side.value,
                    "exit_price": exit_price,
                    "pnl": realized,
                    "reason": "opposite_signal" if signal != pos.side else "flat_signal",
                },
            )

    def _record_trade_pnl(self, symbol: str, pnl: float, trade_pnls: List[float]) -> None:
        idx = self.active_trades.pop(symbol, None)
        if idx is None:
            trade_pnls.append(pnl)
            return
        if idx < len(trade_pnls):
            trade_pnls[idx] = pnl
        else:
            trade_pnls.append(pnl)

    def _build_summary(
        self,
        equity_curve: List[float],
        trade_pnls: List[float],
        risks: List[float],
        *,
        avg_win: float,
        avg_loss: float,
        exposure_pct: float,
        sharpe: float,
    ) -> BacktestSummary:
        ending_equity = self.position_manager.equity
        net_pnl = ending_equity - self.initial_equity
        net_pct = (net_pnl / self.initial_equity * 100) if self.initial_equity else 0.0
        closed_trades = len(trade_pnls)
        win_pct = win_rate(trade_pnls) * 100
        max_dd_pct = abs(max_drawdown(equity_curve)) * 100 if equity_curve else 0.0
        avg_r_multiple = average_r(trade_pnls, risks)
        return BacktestSummary(
            starting_equity=self.initial_equity,
            ending_equity=ending_equity,
            net_pnl=net_pnl,
            net_pnl_pct=net_pct,
            closed_trades=closed_trades,
            win_rate_pct=win_pct,
            max_drawdown_pct=max_dd_pct,
            avg_r_multiple=avg_r_multiple,
            avg_win=avg_win,
            avg_loss=avg_loss,
            exposure_pct=exposure_pct,
            sharpe_ratio=sharpe,
        )

    def _log_equity_snapshot(self, candle: Candle) -> None:
        log_event(
            "EQUITY_SNAPSHOT",
            symbol=candle.symbol,
            mark_price=candle.close,
            equity=self.position_manager.equity,
            balance=self.position_manager.equity,
            unrealized_pnl=0.0,
            open_positions=len(self.position_manager.get_open_positions()),
            run_mode=self.run_mode,
        )
