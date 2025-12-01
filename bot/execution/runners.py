"""Dry-run and shared multi-symbol runner logic."""
from __future__ import annotations

import asyncio
import math
import statistics
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from bot.core.config import BotConfig
from bot.data import fetch_recent_candles
from bot.exchange_info import ExchangeInfoManager
from bot.risk import (
    ExternalSignalGate,
    MultiTimeframeFilter,
    PositionSizer,
    RiskEngine,
    SizingContext,
    TradeEvent,
    volatility_snapshot,
)
from bot.signals import indicators
from bot.signals.strategies import EmaRsiAtrStrategy, StrategyParameters, StrategySignal
from bot.strategy_mode import SCALPING_MODE, build_scalping_config, resolve_strategy_mode
from bot.status import status_store
from bot.utils.logger import get_logger
from core.models import Candle
from strategies.ema_stoch_scalping import EMAStochasticStrategy, ScalpingConfig

logger = get_logger(__name__)

class ScalpingStrategyAdapter:
    def __init__(self, config: ScalpingConfig, symbol: str, interval: str, run_mode: str) -> None:
        self._strategy = EMAStochasticStrategy(config)
        self.symbol = symbol
        self.interval = interval
        self.run_mode = run_mode

    def generate_signals(self, frame: pd.DataFrame) -> List[StrategySignal]:
        candles = self._frame_to_candles(frame)
        if not candles:
            return []
        decision = self._strategy.evaluate(candles)
        snapshot = self._strategy.latest_snapshot or {}
        indicators = dict(decision.indicators)
        indicators.update(
            {
                "size_usd": decision.size_usd,
                "sl_pct": decision.sl_pct,
                "tp_pct": decision.tp_pct,
                "stoch_k_prev": snapshot.get("stoch_k_prev"),
            }
        )
        if decision.action not in {"LONG", "SHORT"}:
            return []
        price = float(candles[-1].close)
        direction = 1 if decision.action == "LONG" else -1
        stop_loss, take_profit = self._compute_levels(price, decision.sl_pct, decision.tp_pct, direction)
        return [
            StrategySignal(
                index=len(frame) - 1,
                direction=direction,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=indicators,
                reason=decision.reason or "ema_stoch_decision",
            )
        ]

    def latest_snapshot(self) -> Optional[Dict[str, float | None]]:
        return self._strategy.latest_snapshot

    @staticmethod
    def _compute_levels(price: float, sl_pct: float, tp_pct: float, direction: int) -> Tuple[float, float]:
        if direction == 1:
            return price * (1 - sl_pct), price * (1 + tp_pct)
        return price * (1 + sl_pct), price * (1 - tp_pct)

    def _frame_to_candles(self, frame: pd.DataFrame) -> List[Candle]:
        candles: List[Candle] = []
        for row in frame.itertuples(index=False):
            open_time = self._to_datetime(getattr(row, "open_time", None))
            close_time = self._to_datetime(getattr(row, "close_time", None)) or open_time
            if open_time is None:
                continue
            candles.append(
                Candle(
                    symbol=self.symbol,
                    open_time=open_time,
                    close_time=close_time or open_time,
                    open=float(getattr(row, "open", 0.0) or 0.0),
                    high=float(getattr(row, "high", 0.0) or 0.0),
                    low=float(getattr(row, "low", 0.0) or 0.0),
                    close=float(getattr(row, "close", 0.0) or 0.0),
                    volume=float(getattr(row, "volume", 0.0) or 0.0),
                )
            )
        return candles

    @staticmethod
    def _to_datetime(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            timestamp = pd.to_datetime(value)
        except Exception:  # noqa: BLE001
            return None
        if pd.isna(timestamp):
            return None
        return timestamp.to_pydatetime()


@dataclass(slots=True)
class PaperPosition:
    symbol: str
    timeframe: str
    direction: int
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    mae: float = 0.0
    mfe: float = 0.0

    def update_extrema(self, mark_price: float) -> None:
        delta = (mark_price - self.entry_price) * self.direction
        pnl = delta * self.quantity
        self.mfe = max(self.mfe, pnl)
        self.mae = min(self.mae, pnl)

    def as_status(self, mark_price: float, pnl: float) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": "LONG" if self.direction == 1 else "SHORT",
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "mark_price": mark_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "pnl": pnl,
            "sizing_mode": self.metadata.get("sizing", {}).get("mode"),
            "volatility": self.metadata.get("sizing", {}).get("volatility"),
            "mta": self.metadata.get("mta"),
            "signals": self.metadata.get("external_signals"),
        }


@dataclass(slots=True)
class SymbolStats:
    symbol: str
    timeframe: str
    trades: int = 0
    wins: int = 0
    realized_pnl: float = 0.0
    last_trade: Optional[Dict[str, Any]] = None

    def record_trade(self, pnl: float, trade_payload: Dict[str, Any]) -> None:
        self.trades += 1
        if pnl > 0:
            self.wins += 1
        self.realized_pnl += pnl
        self.last_trade = trade_payload

    def summary(self, open_pnl: float, has_position: bool) -> Dict[str, Any]:
        win_rate = (self.wins / self.trades) if self.trades else 0.0
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "trades": self.trades,
            "win_rate": win_rate,
            "realized_pnl": self.realized_pnl,
            "open_pnl": open_pnl,
            "has_position": has_position,
        }


@dataclass(slots=True)
class MarketContext:
    symbol: str
    timeframe: str
    params: StrategyParameters
    run_mode: str = "backtest"
    strategy: Optional[Any] = None
    position: Optional[PaperPosition] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    target_notional: Optional[float] = None


class MultiSymbolRunnerBase:
    """Shared lifecycle for paper and live trading loops."""

    def __init__(
        self,
        markets: Sequence[Tuple[str, str, StrategyParameters]],
        exchange_info: ExchangeInfoManager,
        cfg: BotConfig,
        *,
        mode_label: str,
        portfolio_meta: Optional[Dict[str, Any]] = None,
        poll_interval: Optional[int] = None,
        initial_balance: Optional[float] = None,
        status_via_exchange: bool = False,
        risk_engine: Optional[RiskEngine] = None,
        strategy_mode: Optional[str] = None,
    ) -> None:
        if not markets:
            raise ValueError("Multi-symbol runner requires at least one market")
        self._config = cfg
        self._strategy_mode = strategy_mode or resolve_strategy_mode(cfg)
        self._scalping_config = build_scalping_config(cfg) if self._strategy_mode == SCALPING_MODE else None
        self.contexts: List[MarketContext] = [
            MarketContext(symbol=symbol, timeframe=timeframe, params=params, run_mode=cfg.run_mode)
            for symbol, timeframe, params in markets
        ]
        self.exchange_info = exchange_info
        self.mode_label = mode_label
        self.poll_interval = poll_interval or cfg.runtime.poll_interval_seconds
        self.balance = initial_balance if initial_balance is not None else cfg.runtime.paper_account_balance
        self._portfolio_meta = portfolio_meta or {
            "label": f"MULTI ({len(self.contexts)})",
            "symbols": [{"symbol": ctx.symbol, "timeframe": ctx.timeframe} for ctx in self.contexts],
        }
        unique_timeframes = sorted({ctx.timeframe for ctx in self.contexts})
        self._symbol_label = self._portfolio_meta.get("label") or (
            f"MULTI ({len(self.contexts)})" if len(self.contexts) > 1 else self.contexts[0].symbol
        )
        self._timeframe_label = self._portfolio_meta.get("timeframe") or (
            unique_timeframes[0] if len(unique_timeframes) == 1 else "mixed"
        )
        self._stats: Dict[str, SymbolStats] = {
            self._ctx_key(ctx): SymbolStats(ctx.symbol, ctx.timeframe) for ctx in self.contexts
        }
        self._pnl_history: List[float] = []
        self._sizer = PositionSizer(cfg)
        self._multi_filter = MultiTimeframeFilter(cfg)
        self._signal_gate = ExternalSignalGate(cfg)
        self._status_via_exchange = status_via_exchange
        self._risk_engine = risk_engine or RiskEngine(cfg)
        self._last_price_snapshot: Dict[str, float] = {}
        self._risk_state_cache: Optional[Dict[str, Any]] = None
        self._initialize_context_strategies()

    def _initialize_context_strategies(self) -> None:
        for ctx in self.contexts:
            ctx.strategy = self._build_context_strategy(ctx)

    def _build_context_strategy(self, ctx: MarketContext):
        if self._strategy_mode == SCALPING_MODE:
            config = self._scalping_config or build_scalping_config(self._config)
            return ScalpingStrategyAdapter(replace(config), ctx.symbol, ctx.timeframe, ctx.run_mode)
        return EmaRsiAtrStrategy(
            ctx.params,
            symbol=ctx.symbol,
            interval=ctx.timeframe,
            run_mode=ctx.run_mode,
        )

    async def run(self) -> None:
        await self._run_loop(max_cycles=None)

    async def run_cycles(self, cycles: int) -> None:
        if cycles <= 0:
            return
        await self._run_loop(max_cycles=cycles)

    async def _run_loop(self, *, max_cycles: Optional[int]) -> None:
        logger.info("%s runner starting for %s symbols", self.mode_label, len(self.contexts))
        self._initialize_status()
        await self._before_loop()
        completed = 0
        try:
            while True:
                await self._before_step()
                self._step_all()
                completed += 1
                if max_cycles is not None and completed >= max_cycles:
                    break
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            raise
        finally:
            await self._after_loop()
            self._teardown_status()

    def _initialize_status(self) -> None:
        status_store.set_mode(self.mode_label, self._symbol_label, self._timeframe_label)
        status_store.update_balance(self.balance)
        status_store.set_open_pnl(0.0)
        status_store.set_positions([])
        status_store.set_portfolio(self._portfolio_meta)
        status_store.set_symbol_summaries([])
        status_store.set_risk_state(self._risk_engine.snapshot())

    def _teardown_status(self) -> None:
        status_store.set_positions([])
        status_store.set_open_pnl(0.0)
        status_store.set_symbol_summaries([])
        status_store.set_mode("idle", None, None)

    def _step_all(self) -> None:
        self._risk_engine.update_equity(self._equity_for_risk())
        lookback = self._config.runtime.lookback_limit
        for ctx in self.contexts:
            frame = self._fetch_frame(ctx, lookback)
            if frame is None or frame.empty:
                continue
            latest = frame.iloc[-1]
            self._last_price_snapshot[ctx.symbol] = float(latest["close"])
            self._update_position(ctx, latest)
            if ctx.position is None:
                self._maybe_enter(ctx, frame, latest)
        open_positions, total_open_pnl = self._collect_open_positions_snapshot()
        if self._risk_engine.should_flatten_positions():
            reason = self._risk_engine.halt_reason or "daily_loss_limit"
            logger.warning("Force-closing positions due to %s", reason)
            self._force_flatten_positions(reason)
            open_positions, total_open_pnl = self._collect_open_positions_snapshot()
            if self._active_position_count() == 0:
                self._risk_engine.clear_flatten_request()
        if not self._status_via_exchange:
            status_store.set_positions(open_positions)
            status_store.set_open_pnl(total_open_pnl)
            self._publish_symbol_summaries(open_positions)
        self._update_risk_state()

    def _fetch_frame(self, ctx: MarketContext, lookback: int) -> Optional[pd.DataFrame]:
        try:
            candles = fetch_recent_candles(self._config, ctx.symbol, ctx.timeframe, limit=lookback)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch %s %s candles: %s", ctx.symbol, ctx.timeframe, exc)
            return None
        return candles.tail(lookback).reset_index(drop=True)

    def _maybe_enter(self, ctx: MarketContext, frame: pd.DataFrame, latest_row: pd.Series) -> None:
        ctx.target_notional = None
        signals = ctx.strategy.generate_signals(frame)
        if not signals:
            return
        signal = signals[-1]
        if signal.index != len(frame) - 1:
            return
        ctx.target_notional = self._target_notional_from_signal(signal)
        allowed, block_reason = self._risk_engine.can_open_new_trades()
        if not allowed:
            self._log_sizing_skip(ctx, block_reason or "risk_halt")
            return
        price = float(latest_row["close"])
        volatility = volatility_snapshot(frame, self._config)
        balance = self._balance_for_risk()
        sizing_ctx = self._build_sizing_context(ctx, price, volatility, signal.stop_loss)
        if sizing_ctx is None:
            logger.debug("Missing sizing context for %s %s", ctx.symbol, ctx.timeframe)
            return
        decision = self._sizer.plan_trade(sizing_ctx, self._risk_engine)
        if not decision.accepted:
            self._log_sizing_skip(ctx, decision.reason)
            return
        quantity = decision.quantity
        valid, reason, adjusted_qty = self.exchange_info.validate_order(ctx.symbol, price, quantity)
        if not valid or adjusted_qty <= 0:
            log_fn = logger.warning if reason in {"min_qty", "min_notional"} else logger.debug
            log_fn("Rejected %s %s order due to %s", ctx.symbol, ctx.timeframe, reason)
            return
        mta_ok, mta_meta = self._multi_filter.evaluate(ctx.symbol, signal.direction)
        if not mta_ok:
            logger.debug("MTA filter blocked trade for %s %s", ctx.symbol, ctx.timeframe)
            return
        signal_ok, signal_meta = self._signal_gate.evaluate(ctx.symbol, signal.direction)
        if not signal_ok:
            logger.debug("External signal filter blocked trade for %s %s", ctx.symbol, ctx.timeframe)
            return
        opened_at = self._timestamp_from_row(latest_row)
        position = PaperPosition(
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            direction=signal.direction,
            quantity=adjusted_qty,
            entry_price=price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            opened_at=opened_at,
            metadata={"indicators": getattr(signal, "indicators", {})},
        )
        position.metadata.setdefault("sizing", {})
        position.metadata["sizing"].update(
            {
                "mode": self._config.sizing.mode,
                "risk_per_trade": balance * self._config.risk.per_trade_risk,
                "volatility": volatility,
            }
        )
        position.metadata["mta"] = mta_meta
        position.metadata["external_signals"] = signal_meta
        if not self._on_position_open_request(ctx, position, latest_row, signal):
            return
        ctx.position = position
        self._notify_trade_entry()
        logger.info(
            "Opened %s position qty=%.4f entry=%.2f for %s %s",
            "LONG" if signal.direction == 1 else "SHORT",
            adjusted_qty,
            price,
            ctx.symbol,
            ctx.timeframe,
        )

    @staticmethod
    def _target_notional_from_signal(signal: StrategySignal) -> Optional[float]:
        indicators = getattr(signal, "indicators", None) or {}
        raw = indicators.get("size_usd")
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
        return value if value > 0 else None

    def _update_position(self, ctx: MarketContext, row: pd.Series) -> None:
        position = ctx.position
        if not position:
            return
        high = float(row["high"])
        low = float(row["low"])
        exit_price: Optional[float] = None
        exit_reason = "open"
        if position.direction == 1:
            if low <= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = "stop"
            elif high >= position.take_profit:
                exit_price = position.take_profit
                exit_reason = "target"
        else:
            if high >= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = "stop"
            elif low <= position.take_profit:
                exit_price = position.take_profit
                exit_reason = "target"
        if exit_price is None:
            return
        timestamp = self._timestamp_from_row(row)
        self._close_position(ctx, position, exit_price, exit_reason, timestamp)

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
        pnl = (exit_price - position.entry_price) * position.direction * position.quantity
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
        }
        trade_payload.update(position.metadata)
        self._record_trade(ctx, pnl, trade_payload)
        ctx.position = None

    def _record_trade(self, ctx: MarketContext, pnl: float, trade_payload: Dict[str, Any]) -> None:
        stats = self._stats[self._ctx_key(ctx)]
        stats.record_trade(pnl, trade_payload)
        status_store.add_trade(trade_payload)
        if self.mode_label == "live":
            status_store.add_live_trade(trade_payload)
        self._after_trade_closed(ctx, trade_payload)
        trade_timestamp = self._timestamp_to_epoch(trade_payload.get("closed_at"))
        self._risk_engine.register_trade(
            TradeEvent(pnl=pnl, equity=self._equity_for_risk(), timestamp=trade_timestamp, symbol=ctx.symbol)
        )
        self._pnl_history.append(pnl)
        if len(self._pnl_history) > 500:
            self._pnl_history = self._pnl_history[-500:]

    def _publish_symbol_summaries(self, positions: List[Dict[str, Any]]) -> None:
        lookup: Dict[Tuple[str, str], Dict[str, Any]] = {
            (pos["symbol"], pos.get("timeframe", "")): pos for pos in positions
        }
        summaries: List[Dict[str, Any]] = []
        total_trades = 0
        total_wins = 0
        for stats in self._stats.values():
            pos = lookup.get((stats.symbol, stats.timeframe))
            open_pnl = float(pos.get("pnl", 0.0)) if pos else 0.0
            summary = stats.summary(open_pnl=open_pnl, has_position=pos is not None)
            summaries.append(summary)
            total_trades += stats.trades
            total_wins += stats.wins
        summaries.sort(key=lambda item: (item["symbol"], item["timeframe"]))
        status_store.set_symbol_summaries(summaries)
        total_realized = sum(self._pnl_history)
        pnl_wins = [p for p in self._pnl_history if p > 0]
        pnl_losses = [p for p in self._pnl_history if p <= 0]
        if pnl_losses and (loss_sum := sum(pnl_losses)) != 0:
            profit_factor = sum(pnl_wins) / abs(loss_sum)
        else:
            profit_factor = float("inf") if pnl_wins else 0.0
        expectancy = (total_realized / len(self._pnl_history)) if self._pnl_history else 0.0
        sharpe = 0.0
        if len(self._pnl_history) > 1:
            stdev = statistics.pstdev(self._pnl_history)
            if stdev > 0:
                sharpe = statistics.mean(self._pnl_history) / stdev * math.sqrt(len(self._pnl_history))
        metrics = {
            "markets": len(self.contexts),
            "paper_balance": self.balance,
            "open_positions": len(positions),
            "total_realized_pnl": total_realized,
            "win_rate": (total_wins / total_trades) if total_trades else 0.0,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sharpe": sharpe,
        }
        latest_scores = self._signal_gate.latest_scores
        if latest_scores:
            metrics["external_signal"] = statistics.mean(latest_scores.values())
        status_store.set_metrics(metrics)
        if self.mode_label in {"live", "demo-live", "demo-live-smoke"}:
            status_store.set_live_metrics(metrics)

    def _balance_for_risk(self) -> float:
        return self.balance

    def _max_trade_notional(self, ctx: MarketContext, price: float) -> float | None:  # pragma: no cover - override hook
        del price
        hinted = ctx.target_notional
        configured = self._config.sizing.max_notional
        if hinted is not None and hinted > 0:
            if configured is None:
                return hinted
            return min(hinted, configured)
        return configured

    def _build_sizing_context(
        self,
        ctx: MarketContext,
        price: float,
        volatility: Dict[str, float],
        stop_loss: Optional[float],
    ) -> Optional[SizingContext]:
        max_notional = self._max_trade_notional(ctx, price)
        filters = self.exchange_info.get_filters(ctx.symbol)
        symbol_exp, total_exp, active_symbols, symbol_active = self._exposure_for_entry(ctx, price)
        return SizingContext(
            symbol=ctx.symbol,
            balance=self._balance_for_risk(),
            equity=self._equity_for_risk(),
            available_balance=self._available_margin_for_risk(),
            price=price,
            params=ctx.params,
            stop_loss=stop_loss,
            volatility=volatility,
            filters=filters,
            max_notional=max_notional,
            symbol_exposure=symbol_exp,
            total_exposure=total_exp,
            active_symbols=active_symbols,
            symbol_already_active=symbol_active,
        )

    def _equity_for_risk(self) -> float:
        return self.balance

    def _available_margin_for_risk(self) -> float:
        return self.balance

    def _exposure_for_entry(self, ctx: MarketContext, price: float) -> Tuple[float, float, int, bool]:
        return self._paper_exposure_for_entry(ctx, price)

    def _paper_exposure_for_entry(self, ctx: MarketContext, price: float) -> Tuple[float, float, int, bool]:
        total = 0.0
        symbol_total = 0.0
        active_symbols = 0
        for other in self.contexts:
            position = other.position
            if not position or position.quantity <= 0:
                continue
            mark = price if other is ctx else position.entry_price
            notional = abs(position.quantity * max(mark, 0.0))
            if notional <= 0:
                continue
            total += notional
            if notional > 0:
                active_symbols += 1
            if other.symbol == ctx.symbol:
                symbol_total += notional
        return symbol_total, total, active_symbols, symbol_total > 0

    def _collect_open_positions_snapshot(self) -> Tuple[List[Dict[str, Any]], float]:
        open_positions: List[Dict[str, Any]] = []
        total_open_pnl = 0.0
        for ctx in self.contexts:
            position = ctx.position
            if not position or position.quantity <= 0:
                continue
            mark_price = self._last_price_snapshot.get(ctx.symbol, position.entry_price)
            pnl = (mark_price - position.entry_price) * position.direction * position.quantity
            position.update_extrema(mark_price)
            open_positions.append(position.as_status(mark_price, pnl))
            total_open_pnl += pnl
        return open_positions, total_open_pnl

    def _active_position_count(self) -> int:
        return sum(1 for ctx in self.contexts if ctx.position and ctx.position.quantity > 0)

    def _notify_trade_entry(self) -> None:
        if self._risk_engine:
            self._risk_engine.record_trade_entry()

    def _force_flatten_positions(self, reason: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        for ctx in self.contexts:
            position = ctx.position
            if not position or position.quantity <= 0:
                continue
            mark_price = self._last_price_snapshot.get(ctx.symbol, position.entry_price)
            self._close_position(ctx, position, mark_price, reason, timestamp)

    def _log_sizing_skip(self, ctx: MarketContext, reason: Optional[str]) -> None:
        message = f"Sizing rejected for {ctx.symbol} {ctx.timeframe}: {reason or 'unknown'}"
        log_fn = logger.info if self.mode_label == "live" else logger.debug
        log_fn(message)

    def _timestamp_from_row(self, row: pd.Series) -> str:
        timestamp = row.get("close_time") or row.get("open_time")
        if hasattr(timestamp, "isoformat"):
            return str(timestamp)
        return datetime.utcnow().isoformat()

    @staticmethod
    def _timestamp_to_epoch(raw: Any) -> Optional[float]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            return value if value > 0 else None
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        return None

    def _ctx_key(self, ctx: MarketContext) -> str:
        return f"{ctx.symbol}:{ctx.timeframe}"

    async def _before_loop(self) -> None:  # pragma: no cover - override hook
        return None

    async def _after_loop(self) -> None:  # pragma: no cover - override hook
        return None

    async def _before_step(self) -> None:  # pragma: no cover - override hook
        return None

    def _on_position_open_request(
        self,
        ctx: MarketContext,
        position: PaperPosition,
        latest_row: pd.Series,
        signal: StrategySignal,
    ) -> bool:  # pragma: no cover - override hook
        del ctx, position, latest_row, signal
        return True

    def _on_position_close_request(
        self,
        ctx: MarketContext,
        position: PaperPosition,
        exit_price: float,
        exit_reason: str,
    ) -> bool:  # pragma: no cover - override hook
        del ctx, position, exit_price, exit_reason
        return True

    def _after_trade_closed(self, ctx: MarketContext, trade_payload: Dict[str, Any]) -> None:  # pragma: no cover
        del ctx, trade_payload
        return None

    def _update_risk_state(self) -> None:
        snapshot = self._risk_engine.snapshot()
        previous = self._risk_state_cache or {}
        halted_changed = bool(snapshot.get("trading_paused")) != bool(previous.get("trading_paused"))
        reason_changed = snapshot.get("reason") != previous.get("reason")
        if halted_changed or reason_changed:
            state = "PAUSED" if snapshot.get("trading_paused") else "ACTIVE"
            logger.warning(
                "Risk engine state changed -> %s (reason=%s, loss%%=%.4f, loss_abs=%.2f)",
                state,
                snapshot.get("reason"),
                float(snapshot.get("loss_pct", 0.0) or 0.0),
                float(snapshot.get("loss_abs", 0.0) or 0.0),
            )
        status_store.set_risk_state(snapshot)
        self._risk_state_cache = snapshot


class MultiSymbolDryRunner(MultiSymbolRunnerBase):
    def __init__(
        self,
        markets: Sequence[Tuple[str, str, StrategyParameters]],
        exchange_info: ExchangeInfoManager,
        cfg: BotConfig,
        *,
        mode_label: str = "dry_run_multi",
        portfolio_meta: Optional[Dict[str, Any]] = None,
        strategy_mode: Optional[str] = None,
    ) -> None:
        super().__init__(
            markets,
            exchange_info,
            cfg,
            mode_label=mode_label,
            portfolio_meta=portfolio_meta,
            strategy_mode=strategy_mode,
        )


class DryRunner(MultiSymbolDryRunner):
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        exchange_info: ExchangeInfoManager,
        params: StrategyParameters,
        strategy: Optional[Any] = None
    ) -> None:
        markets = [(symbol, timeframe, params)]
        portfolio = {
            "label": symbol,
            "timeframe": timeframe,
            "symbols": [{"symbol": symbol, "timeframe": timeframe}],
        }
        super().__init__(
            markets,
            exchange_info,
            cfg,
            mode_label="dry_run",
            portfolio_meta=portfolio,
            strategy_mode=resolve_strategy_mode(cfg),
        )


__all__ = [
    "DryRunner",
    "MultiSymbolDryRunner",
    "MultiSymbolRunnerBase",
    "PaperPosition",
    "SymbolStats",
    "MarketContext",
]
