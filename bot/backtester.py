"""Backtesting engine that reuses the live multi-symbol runner."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bot.core.config import BotConfig
from bot.data.feeds import validate_candles
from bot.exchange_info import ExchangeInfoManager
from bot.execution.runners import MarketContext, MultiSymbolRunnerBase, PaperPosition
from bot.signals.strategies import StrategyParameters
from bot.status import status_store
from bot.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RealismProfile:
    slippage_scale: float = 1.0
    fee_scale: float = 1.0
    latency_scale: float = 1.0
    force_latency: Optional[bool] = None
    force_funding: Optional[bool] = None


REALISM_PROFILES: Dict[str, RealismProfile] = {
    "toy": RealismProfile(slippage_scale=0.5, fee_scale=0.5, latency_scale=0.0, force_latency=False, force_funding=False),
    "standard": RealismProfile(),
    "aggressive": RealismProfile(
        slippage_scale=2.0,
        fee_scale=1.25,
        latency_scale=1.5,
        force_latency=True,
        force_funding=True,
    ),
}


def _timeframe_to_minutes(timeframe: str) -> float:
    if not timeframe:
        return 1.0
    unit = timeframe[-1].lower()
    try:
        value = float(timeframe[:-1])
    except ValueError:
        return 1.0
    unit_map = {"m": 1.0, "h": 60.0, "d": 1_440.0, "w": 10_080.0}
    multiplier = unit_map.get(unit)
    if multiplier is None:
        return 1.0
    minutes = max(value * multiplier, 1.0)
    return minutes


@dataclass(slots=True)
class TradeResult:
    symbol: str
    timeframe: str
    direction: int
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    reason: str
    opened_at: str
    closed_at: str
    fees: float
    entry_index: Optional[int] = None
    exit_index: Optional[int] = None
    funding: float = 0.0


class BacktestRunner(MultiSymbolRunnerBase):
    def __init__(
        self,
        cfg: BotConfig,
        symbol: str,
        timeframe: str,
        candles: pd.DataFrame,
        params: StrategyParameters,
        exchange_info: ExchangeInfoManager,
    ) -> None:
        self._config = cfg
        validate_candles(candles, symbol, timeframe)
        dataset = candles.tail(cfg.backtest.max_bars).reset_index(drop=True)
        markets = [(symbol, timeframe, params)]
        super().__init__(
            markets,
            exchange_info,
            cfg,
            mode_label="backtest",
            initial_balance=cfg.backtest.initial_balance,
        )
        self._symbol = symbol
        self._timeframe = timeframe
        self._dataset = dataset
        self._cursor = 0
        self._timeframe_minutes = _timeframe_to_minutes(timeframe)
        self._timeframe_ms = self._timeframe_minutes * 60_000
        profile = REALISM_PROFILES.get(cfg.backtest.realism_level, REALISM_PROFILES["standard"])
        base_fee = cfg.risk.taker_fee if cfg.backtest.fee_model == "taker" else cfg.risk.maker_fee
        self._fee_rate = base_fee * profile.fee_scale
        base_slippage = cfg.backtest.slippage_bps / 10_000
        self._slippage = base_slippage * profile.slippage_scale
        latency_flag = profile.force_latency if profile.force_latency is not None else cfg.backtest.enable_latency
        self._latency_enabled = bool(latency_flag)
        latency_ms = cfg.backtest.latency_ms * profile.latency_scale if self._latency_enabled else 0.0
        self._latency_ms = latency_ms
        self._latency_ratio = self._compute_latency_ratio()
        funding_flag = profile.force_funding if profile.force_funding is not None else cfg.backtest.enable_funding_costs
        self._funding_enabled = bool(funding_flag)
        self._funding_rate = cfg.backtest.funding_rate_bps / 10_000
        self._funding_bars = self._compute_funding_bars()
        self._bars_since_funding = 0
        self._trade_log: List[TradeResult] = []

    def execute(self) -> Dict[str, object]:
        total_rows = len(self._dataset)
        if total_rows == 0:
            logger.warning("Backtest dataset empty for %s %s", self._symbol, self._timeframe)
            return {"symbol": self._symbol, "timeframe": self._timeframe, "trades": [], "metrics": {}}
        status_store.set_mode("backtest", self._symbol, self._timeframe)
        status_store.update_balance(self.balance)
        status_store.set_positions([])
        status_store.set_open_pnl(0.0)
        while self._cursor < total_rows:
            self._cursor += 1
            self._step_all()
            self._after_bar_advanced()
        status_store.set_mode("idle", None, None)
        metrics = self._compute_metrics()
        return {"symbol": self._symbol, "timeframe": self._timeframe, "trades": self._trade_log, "metrics": metrics}

    def _fetch_frame(self, ctx: MarketContext, lookback: int) -> Optional[pd.DataFrame]:  # type: ignore[override]
        del ctx
        end = self._cursor
        if end <= 0:
            return None
        start = max(0, end - lookback)
        frame = self._dataset.iloc[start:end]
        if frame.empty:
            return None
        return frame.reset_index(drop=True)

    def _on_position_open_request(
        self,
        ctx: MarketContext,
        position: PaperPosition,
        latest_row: pd.Series,
        signal,
    ) -> bool:  # type: ignore[override]
        del ctx, signal
        slippage = position.entry_price * self._slippage
        adjustment = slippage if position.direction == 1 else -slippage
        position.entry_price += adjustment
        entry_fee = abs(position.entry_price * position.quantity) * self._fee_rate
        position.metadata.setdefault("fees", 0.0)
        position.metadata["fees"] += entry_fee
        position.metadata["entry_index"] = max(self._cursor - 1, 0)
        if self._latency_ratio > 0:
            self._apply_latency_penalty(position, latest_row)
            position.metadata["latency_ms"] = self._latency_ms
        return True

    def _on_position_close_request(
        self,
        ctx: MarketContext,
        position: PaperPosition,
        exit_price: float,
        exit_reason: str,
    ) -> bool:  # type: ignore[override]
        del ctx, exit_reason
        exit_fee = abs(exit_price * position.quantity) * self._fee_rate
        position.metadata.setdefault("fees", 0.0)
        position.metadata["fees"] += exit_fee
        position.metadata["exit_index"] = max(self._cursor - 1, 0)
        return True

    def _close_position(
        self,
        ctx: MarketContext,
        position: PaperPosition,
        exit_price: float,
        exit_reason: str,
        closed_at: str,
    ) -> None:
        if not self._on_position_close_request(ctx, position, exit_price, exit_reason):
            return
        if self._latency_ratio > 0:
            exit_price = self._apply_exit_latency_penalty(position, exit_price)
        slip = exit_price * self._slippage
        exit_price = exit_price - slip if position.direction == 1 else exit_price + slip
        fees = float(position.metadata.get("fees", 0.0) or 0.0)
        pnl = (exit_price - position.entry_price) * position.direction * position.quantity - fees
        self.balance += pnl
        status_store.update_balance(self.balance)
        trade_payload = {
            "mode": self.mode_label,
            "symbol": ctx.symbol,
            "timeframe": ctx.timeframe,
            "side": "LONG" if position.direction == 1 else "SHORT",
            "quantity": position.quantity,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": exit_reason,
            "opened_at": position.opened_at,
            "closed_at": closed_at,
            "mae": position.mae,
            "mfe": position.mfe,
            "fees": fees,
            "entry_index": position.metadata.get("entry_index"),
            "exit_index": position.metadata.get("exit_index"),
        }
        trade_payload.update({k: v for k, v in position.metadata.items() if k not in {"fees"}})
        self._record_trade(ctx, pnl, trade_payload)
        ctx.position = None

    def _record_trade(self, ctx: MarketContext, pnl: float, trade_payload: Dict[str, object]) -> None:
        super()._record_trade(ctx, pnl, trade_payload)
        result = TradeResult(
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            direction=1 if trade_payload.get("side") == "LONG" else -1,
            quantity=float(trade_payload.get("quantity", 0.0)),
            entry_price=float(trade_payload.get("entry_price", 0.0)),
            exit_price=float(trade_payload.get("exit_price", 0.0)),
            pnl=float(trade_payload.get("pnl", 0.0)),
            reason=str(trade_payload.get("reason", "")),
            opened_at=str(trade_payload.get("opened_at", "")),
            closed_at=str(trade_payload.get("closed_at", "")),
            fees=float(trade_payload.get("fees", 0.0)),
            entry_index=int(trade_payload.get("entry_index") or 0),
            exit_index=int(trade_payload.get("exit_index") or 0),
            funding=float(trade_payload.get("funding_paid", 0.0)),
        )
        self._trade_log.append(result)

    def _compute_metrics(self) -> Dict[str, float]:
        pnl = np.array([trade.pnl for trade in self._trade_log]) if self._trade_log else np.array([])
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        total = float(pnl.sum()) if pnl.size else 0.0
        win_rate = float(len(wins) / len(self._trade_log)) if self._trade_log else 0.0
        if losses.size and losses.sum() != 0:
            profit_factor = float(wins.sum() / abs(losses.sum()))
        else:
            profit_factor = float("inf") if wins.size else 0.0
        expectancy = total / len(self._trade_log) if self._trade_log else 0.0
        equity_curve = self._build_equity_curve()
        sharpe, sortino, max_drawdown = self._risk_metrics(equity_curve)
        return {
            "total_pnl": total,
            "trades": len(self._trade_log),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "ending_balance": equity_curve[-1] if equity_curve else self._config.backtest.initial_balance,
        }

    def _build_equity_curve(self) -> List[float]:
        equity = self._config.backtest.initial_balance
        curve = [equity]
        for trade in self._trade_log:
            equity += trade.pnl
            curve.append(equity)
        return curve

    def _risk_metrics(self, equity_curve: List[float]) -> tuple[float, float, float]:
        if len(equity_curve) < 2:
            return 0.0, 0.0, 0.0
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0.0
        downside = returns[returns < 0]
        sortino = float(returns.mean() / downside.std() * np.sqrt(252)) if downside.size and downside.std() != 0 else 0.0
        peak = -np.inf
        max_drawdown = 0.0
        for value in equity_curve:
            peak = max(peak, value)
            if peak:
                max_drawdown = max(max_drawdown, (peak - value) / peak)
        return sharpe, sortino, float(max_drawdown)

    def _after_bar_advanced(self) -> None:
        if not self._funding_enabled or self._funding_bars <= 0:
            return
        self._bars_since_funding += 1
        if self._bars_since_funding < self._funding_bars:
            return
        self._bars_since_funding = 0
        self._apply_funding_payments()

    def _apply_latency_penalty(self, position: PaperPosition, latest_row: pd.Series) -> None:
        if self._latency_ratio <= 0:
            return
        high = float(latest_row.get("high", position.entry_price))
        low = float(latest_row.get("low", position.entry_price))
        if position.direction == 1:
            drift = max(0.0, high - position.entry_price) * self._latency_ratio
            position.entry_price += drift
        else:
            drift = max(0.0, position.entry_price - low) * self._latency_ratio
            position.entry_price -= drift

    def _apply_funding_payments(self) -> None:
        total_adjustment = 0.0
        for ctx in self.contexts:
            position = ctx.position
            if not position or position.quantity <= 0:
                continue
            notional = abs(position.entry_price * position.quantity)
            if notional <= 0:
                continue
            payment = notional * self._funding_rate
            adjustment = payment if position.direction == 1 else -payment
            position.metadata.setdefault("fees", 0.0)
            position.metadata.setdefault("funding_paid", 0.0)
            position.metadata["fees"] += adjustment
            position.metadata["funding_paid"] += adjustment
            total_adjustment += adjustment
            logger.debug(
                "Applied funding %.4f to %s %s (%s)",
                adjustment,
                ctx.symbol,
                ctx.timeframe,
                "LONG" if position.direction == 1 else "SHORT",
            )
        if total_adjustment == 0:
            return
        self.balance -= total_adjustment
        status_store.update_balance(self.balance)
        self._risk_engine.update_equity(self._equity_for_risk())
        realized = -total_adjustment
        if realized != 0:
            self._risk_engine.register_trade(realized, equity=self._equity_for_risk(), timestamp=time.time())

    def _compute_latency_ratio(self) -> float:
        if not self._latency_enabled or self._timeframe_ms <= 0:
            return 0.0
        if self._latency_ms <= 0:
            return 0.0
        ratio = min(self._latency_ms / self._timeframe_ms, 1.0)
        return max(ratio, 0.0)

    def _compute_funding_bars(self) -> int:
        if not self._funding_enabled:
            return 0
        if self._timeframe_minutes <= 0:
            return 0
        interval_minutes = max(self._config.backtest.funding_interval_hours * 60.0, self._timeframe_minutes)
        bars = int(round(interval_minutes / self._timeframe_minutes))
        return max(bars, 1)

    def _apply_exit_latency_penalty(self, position: PaperPosition, exit_price: float) -> float:
        if self._latency_ratio <= 0:
            return exit_price
        row = self._latency_reference_row()
        if row is not None:
            high = float(row.get("high", exit_price))
            low = float(row.get("low", exit_price))
            if position.direction == 1:
                adverse = min(exit_price, low)
                drift = max(0.0, exit_price - adverse)
                return exit_price - drift * self._latency_ratio
            adverse = max(exit_price, high)
            drift = max(0.0, adverse - exit_price)
            return exit_price + drift * self._latency_ratio
        penalty = exit_price * 0.0005 * self._latency_ratio
        return exit_price - penalty if position.direction == 1 else exit_price + penalty

    def _latency_reference_row(self) -> Optional[pd.Series]:
        if self._cursor <= 0:
            return None
        index = min(self._cursor - 1, len(self._dataset) - 1)
        if index < 0:
            return None
        return self._dataset.iloc[index]


class Backtester:
    def __init__(self, cfg: BotConfig, exchange_info: ExchangeInfoManager) -> None:
        self._config = cfg
        self.exchange_info = exchange_info

    def run(
        self,
        symbol: str,
        timeframe: str,
        candles: pd.DataFrame,
        params: StrategyParameters,
    ) -> Dict[str, object]:
        runner = BacktestRunner(self._config, symbol, timeframe, candles, params, self.exchange_info)
        return runner.execute()


__all__ = ["Backtester", "BacktestRunner", "TradeResult"]
