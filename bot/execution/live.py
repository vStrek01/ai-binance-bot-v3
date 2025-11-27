"""Live trading runner that extends the multi-symbol loop with testnet orders."""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bot.core.config import BotConfig
from bot.exchange_info import ExchangeInfoManager
from bot.execution.balance_manager import BalanceManager, BalanceSnapshot, ExchangePosition
from bot.execution.exchange_client import ExchangeClient, ExchangeRequestError
from bot.execution.market_loop import EntryPlan, ExitPlan, MarketLoop
from bot.execution.order_manager import OrderManager, OrderPlacementError, OrderRequest
from bot.execution.risk_gate import RiskGate
from bot.learning import TradeLearningStore
from bot.live_logging import LiveTradeLogger, OrderAuditLogger
from bot.execution.runners import MarketContext, MultiSymbolRunnerBase, PaperPosition
from bot.risk import RiskEngine, TradeEvent, volatility_snapshot
from bot.signals.strategies import StrategyParameters, StrategySignal
from bot.status import status_store
from bot.utils.logger import get_logger
from infra.logging import log_event

logger = get_logger(__name__)


def resolve_learning_store(cfg: BotConfig, store: Optional[TradeLearningStore]) -> Optional[TradeLearningStore]:
    if not cfg.runtime.use_learning_store:
        return None
    if store is not None:
        return store
    return TradeLearningStore(cfg)


class LiveTrader(MultiSymbolRunnerBase):

    def __init__(
        self,
        markets: Sequence[Tuple[str, str, StrategyParameters]],
        exchange_info: ExchangeInfoManager,
        cfg: BotConfig,
        *,
        client: ExchangeClient,
        portfolio_meta: Optional[Dict[str, Any]] = None,
        learning_store: Optional[TradeLearningStore] = None,
        mode_label: str = "live",
    ) -> None:
        self._config = cfg
        self._account_asset = cfg.runtime.demo_account_asset.upper()
        self.client = client
        self.order_logger = OrderAuditLogger(cfg)
        self.trade_logger = LiveTradeLogger(cfg)
        self.learning_store = resolve_learning_store(cfg, learning_store)
        self._applied_learning: Dict[str, Dict[str, Any]] = {}
        self._last_trade_refresh = 0.0
        self._last_position_snapshot: List[Dict[str, Any]] = []
        self._last_trade_ids: Dict[str, int] = {}
        self._synced_order_ids: set[int] = set()
        self._session_start_ms = int(time.time() * 1000)
        self._latest_frames: Dict[str, Any] = {}
        self._last_learning_update: Dict[str, float] = {}
        self._risk_engine = RiskEngine(cfg)
        self.balance_manager = BalanceManager(cfg, client=client, risk_engine=self._risk_engine, exchange_info=exchange_info)
        self.order_manager = OrderManager(cfg, client=self.client, logger=self.order_logger, balance_manager=self.balance_manager)
        self._risk_gate = RiskGate(cfg, self._risk_engine)
        self._pending_order_requests: Dict[str, OrderRequest] = {}
        balance_snapshot = self.balance_manager.refresh_account_balance(force=True)
        self.total_balance = balance_snapshot.total
        self.available_balance = balance_snapshot.available
        self.balance = self.available_balance
        initial_balance = self.available_balance
        super().__init__(
            markets,
            exchange_info,
            cfg,
            mode_label=mode_label,
            portfolio_meta=portfolio_meta,
            initial_balance=initial_balance,
            status_via_exchange=True,
            risk_engine=self._risk_engine,
        )
        self._ctx_lookup: Dict[str, MarketContext] = {ctx.symbol: ctx for ctx in self.contexts}
        self._trade_refresh_interval = max(5.0, min(float(self.poll_interval) * 2.0, 30.0))
        self._market_loops: Dict[str, MarketLoop] = {
            ctx.symbol: MarketLoop(
                ctx,
                cfg,
                risk_gate=self._risk_gate,
                sizer=self._sizer,
                multi_filter=self._multi_filter,
                external_gate=self._signal_gate,
                exchange_info=self.exchange_info,
                sizing_builder=self._build_sizing_context,
                timestamp_fn=self._timestamp_from_row,
                log_sizing_skip=self._log_sizing_skip,
            )
            for ctx in self.contexts
        }
        self._position_cache: Dict[str, Dict[str, Any]] = {}
        self._pending_closures: set[str] = set()
        self._kill_switch_event_key: Optional[Tuple[str, float]] = None

    async def _before_loop(self) -> None:
        for ctx in self.contexts:
            self._ensure_leverage(ctx.symbol)
        self._sync_live_positions_and_trades(force=True, backfill_trades=True)
        await super()._before_loop()

    async def _before_step(self) -> None:
        self._refresh_account_balance()
        self._apply_learned_parameters()
        await super()._before_step()

    def _step_all(self) -> None:
        self._sync_live_positions_and_trades()
        super()._step_all()
        self._sync_live_positions_and_trades()

    def _fetch_frame(self, ctx: MarketContext, lookback: int):  # type: ignore[override]
        frame = super()._fetch_frame(ctx, lookback)
        if frame is not None and not frame.empty:
            self._latest_frames[ctx.symbol] = frame
        return frame

    def _maybe_enter(self, ctx: MarketContext, frame, latest_row):  # type: ignore[override]
        loop = self._market_loops.get(ctx.symbol)
        if not loop:
            return
        exposure = self._risk_engine.compute_exposure(self.balance_manager.exposure_payload())
        plan = loop.plan_entry(
            frame,
            latest_row,
            balance=self.balance,
            equity=self._estimate_equity(),
            available_balance=self.balance_manager.available_balance,
            exposure=exposure,
        )
        if not plan:
            return
        self._pending_order_requests[ctx.symbol] = plan.order_request
        success = self._on_position_open_request(ctx, plan.position, latest_row, plan.signal, plan.order_request)
        self._pending_order_requests.pop(ctx.symbol, None)
        if not success:
            return
        ctx.position = plan.position
        logger.info(
            "Opened %s position qty=%.4f entry=%.2f for %s %s",
            "LONG" if plan.signal.direction == 1 else "SHORT",
            plan.position.quantity,
            plan.position.entry_price,
            ctx.symbol,
            ctx.timeframe,
        )

    def _update_position(self, ctx: MarketContext, row):  # type: ignore[override]
        position = ctx.position
        if not position:
            return
        loop = self._market_loops.get(ctx.symbol)
        if not loop:
            return
        plan = loop.plan_exit(position, row)
        if not plan:
            return
        filters = self.exchange_info.get_filters(ctx.symbol)
        order_request = OrderRequest(
            symbol=ctx.symbol,
            side=plan.side,
            quantity=plan.quantity,
            order_type="MARKET",
            reduce_only=True,
            price=plan.price,
            filters=filters,
            tag=plan.reason,
        )
        self._pending_order_requests[ctx.symbol] = order_request
        timestamp = self._timestamp_from_row(row)
        self._close_position(ctx, position, plan.price, plan.reason, timestamp)
        self._pending_order_requests.pop(ctx.symbol, None)

    def _balance_for_risk(self) -> float:
        return self._refresh_account_balance()

    def _ensure_leverage(self, symbol: str) -> None:
        try:
            self.client.change_leverage(symbol=symbol, leverage=int(self._config.risk.leverage))
        except ExchangeRequestError as exc:
            logger.warning("Unable to set leverage for %s: %s", symbol, exc)

    def _refresh_account_balance(self, force: bool = False) -> float:
        snapshot = self.balance_manager.refresh_account_balance(force=force)
        self.total_balance = snapshot.total
        self.available_balance = snapshot.available
        self.balance = snapshot.available
        status_store.update_balance(snapshot.available)
        self._emit_equity_event(snapshot)
        return snapshot.available


    def _on_position_open_request(
        self,
        ctx: MarketContext,
        position: PaperPosition,
        latest_row: Any,
        signal: StrategySignal,
        order_request: OrderRequest | None = None,
    ) -> bool:
        del latest_row, signal
        request = order_request or self._pending_order_requests.pop(ctx.symbol, None)
        if request is None:
            side = "BUY" if position.direction == 1 else "SELL"
            filters = self.exchange_info.get_filters(ctx.symbol)
            request = OrderRequest(
                symbol=ctx.symbol,
                side=side,
                quantity=position.quantity,
                order_type="MARKET",
                reduce_only=False,
                price=position.entry_price,
                filters=filters,
            )
        try:
            placed = self.order_manager.submit_order(request)
        except OrderPlacementError as exc:
            logger.warning("Entry order failed for %s: %s", ctx.symbol, exc)
            return False
        if placed.price is not None:
            position.entry_price = placed.price
        if placed.quantity > 0:
            position.quantity = placed.quantity
        position.metadata.setdefault("indicators", {})
        position.metadata["entry_order_id"] = placed.order_id
        position.metadata["opened_at"] = self._timestamp_from_response(placed.raw)
        self._record_position_open(ctx, position, source=(request.tag or "entry"))
        return True

    def _on_position_close_request(
        self,
        ctx: MarketContext,
        position: PaperPosition,
        exit_price: float,
        exit_reason: str,
        order_request: OrderRequest | None = None,
    ) -> bool:
        del exit_reason
        self.balance_manager.sync_positions(force=True)
        side = "SELL" if position.direction == 1 else "BUY"
        live_qty = self.balance_manager.live_position_quantity(ctx.symbol, position.direction)
        if live_qty <= 0:
            self._clear_local_position(ctx, reason="exchange_flat")
            logger.info("Skipping reduce-only close for %s %s: no exchange position", ctx.symbol, ctx.timeframe)
            return False
        if position.quantity > live_qty:
            position.quantity = live_qty
        request = order_request or self._pending_order_requests.pop(ctx.symbol, None)
        if request is None:
            filters = self.exchange_info.get_filters(ctx.symbol)
            request = OrderRequest(
                symbol=ctx.symbol,
                side=side,
                quantity=position.quantity,
                order_type="MARKET",
                reduce_only=True,
                price=exit_price,
                filters=filters,
                tag="exit",
            )
        try:
            placed = self.order_manager.submit_order(request)
        except OrderPlacementError as exc:
            logger.warning("Reduce-only close failed for %s: %s", ctx.symbol, exc)
            return False
        position.metadata["exit_order_id"] = placed.order_id
        position.metadata["closed_at"] = self._timestamp_from_response(placed.raw)
        self._reconcile_after_reduce_only(ctx, position.direction)
        return True

    def _after_trade_closed(self, ctx: MarketContext, trade_payload: Dict[str, Any]) -> None:
        self.trade_logger.log(trade_payload)
        if self.learning_store is not None:
            self.learning_store.record_trade(trade_payload, ctx.params)
        exit_order_id = self._int_or_none(trade_payload.get("exit_order_id"))
        if exit_order_id is not None:
            self._synced_order_ids.add(exit_order_id)
        self._record_position_close(ctx, trade_payload)

    def _apply_learned_parameters(self) -> None:
        if self.learning_store is None:
            return
        for ctx in self.contexts:
            overrides = self.learning_store.best_params(ctx.symbol, ctx.timeframe)
            if not overrides:
                continue
            key = self._ctx_key(ctx)
            trade_samples = self.learning_store.trade_count(ctx.symbol, ctx.timeframe)
            if trade_samples < self._config.reinforcement.min_trades_before_update:
                continue
            now = time.time()
            last_update = self._last_learning_update.get(key, 0.0)
            if (now - last_update) < self._config.reinforcement.update_cooldown_seconds:
                continue
            cached = self._applied_learning.get(key)
            if cached == overrides:
                continue
            new_params = self._merge_parameters(ctx.params, overrides)
            if new_params == ctx.params:
                self._applied_learning[key] = overrides
                continue
            ctx.params = new_params
            ctx.strategy = ctx.strategy.__class__(ctx.params)
            self._applied_learning[key] = overrides
            self._last_learning_update[key] = now
            logger.info(
                "Updated parameters for %s %s based on last %s demo trades",
                ctx.symbol,
                ctx.timeframe,
                trade_samples,
            )

    @staticmethod
    def _merge_parameters(current: StrategyParameters, overrides: Dict[str, Any]) -> StrategyParameters:
        def _ival(key: str, fallback: int) -> int:
            raw = overrides.get(key, fallback)
            try:
                return int(float(raw))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return fallback

        def _fval(key: str, fallback: float) -> float:
            raw = overrides.get(key, fallback)
            try:
                value = float(raw)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                value = fallback
            if math.isnan(value) or math.isinf(value):
                return fallback
            return value

        def _clamp(name: str, value: float) -> float:
            bounds = self._config.reinforcement.hard_parameter_bounds.get(name)
            if not bounds:
                return value
            low, high = bounds
            return max(low, min(high, value))

        fast_ema = int(round(_clamp("fast_ema", float(_ival("fast_ema", current.fast_ema)))))
        slow_ema = int(round(_clamp("slow_ema", float(_ival("slow_ema", current.slow_ema)))))
        rsi_length = int(round(_clamp("rsi_length", float(_ival("rsi_length", current.rsi_length)))))
        cooldown_bars = int(round(_clamp("cooldown_bars", float(_ival("cooldown_bars", current.cooldown_bars)))))
        hold_bars = int(round(_clamp("hold_bars", float(_ival("hold_bars", current.hold_bars)))))
        rsi_overbought = _clamp("rsi_overbought", _fval("rsi_overbought", current.rsi_overbought))
        rsi_oversold = _clamp("rsi_oversold", _fval("rsi_oversold", current.rsi_oversold))
        atr_stop = _clamp("atr_stop", _fval("atr_stop", current.atr_stop))
        atr_target = _clamp("atr_target", _fval("atr_target", current.atr_target))
        min_volatility_pct = _fval("min_volatility_pct", current.min_volatility_pct)
        max_spread_pct = _fval("max_spread_pct", current.max_spread_pct)
        trend_confirm_bars = _ival("trend_confirm_bars", current.trend_confirm_bars)

        return StrategyParameters(
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            rsi_length=rsi_length,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            atr_period=_ival("atr_period", current.atr_period),
            atr_stop=atr_stop,
            atr_target=atr_target,
            cooldown_bars=cooldown_bars,
            hold_bars=hold_bars,
            min_volatility_pct=min_volatility_pct,
            max_spread_pct=max_spread_pct,
            trend_confirm_bars=trend_confirm_bars,
        )

    @staticmethod
    def _parse_price(response: Dict[str, Any]) -> Optional[float]:
        for key in ("avgPrice", "price", "executedPrice"):
            raw = response.get(key)
            if raw:
                try:
                    value = float(raw)
                    if value > 0:
                        return value
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    continue
        return None

    @staticmethod
    def _timestamp_from_response(response: Dict[str, Any]) -> str:
        for key in ("updateTime", "transactTime", "timestamp"):
            raw = response.get(key)
            if raw:
                try:
                    value = float(raw)
                    if value > 1_000_000_000_000:  # likely milliseconds
                        value /= 1000
                    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))
                except (TypeError, ValueError):  # pragma: no cover
                    continue
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _max_trade_notional(self, ctx: MarketContext, price: float) -> float | None:
        filters = self.exchange_info.get_filters(ctx.symbol)
        available = max(self.balance_manager.available_balance, 0.0)
        if price <= 0 or available <= 0:
            return 0.0
        leverage_cap = filters.max_leverage if filters else self._config.risk.leverage
        leverage = min(self._config.risk.leverage, leverage_cap)
        safety = max(min(self._config.runtime.max_margin_utilization, 0.99), 0.1)
        max_notional = available * leverage * safety
        if filters and max_notional < filters.min_notional:
            logger.warning(
                "Available margin %.2f %s insufficient for %s (min %.2f)",
                available,
                self._account_asset,
                ctx.symbol,
                filters.min_notional,
            )
            return 0.0
        return max_notional

    def _log_sizing_skip(self, ctx: MarketContext, reason: Optional[str]) -> None:
        if reason == "margin" and not self._risk_engine.should_log_margin_block(ctx.symbol, self.balance_manager.available_balance):
            return
        super()._log_sizing_skip(ctx, reason)

    def _equity_for_risk(self) -> float:
        return self._estimate_equity()

    def _available_margin_for_risk(self) -> float:
        return max(self.balance_manager.available_balance, 0.0)

    def _exposure_for_entry(self, ctx: MarketContext, price: float) -> Tuple[float, float, int, bool]:
        state = self._risk_engine.compute_exposure(self.balance_manager.exposure_payload())
        symbol_total = state.per_symbol.get(ctx.symbol, 0.0)
        active_symbols = sum(1 for value in state.per_symbol.values() if value > 0)
        return symbol_total, state.total, active_symbols, symbol_total > 0

    def _estimate_equity(self) -> float:
        return self.balance_manager.estimate_equity()

    @staticmethod
    def _safe_float(value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
            if math.isnan(parsed) or math.isinf(parsed):
                return fallback
            return parsed
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return fallback

    def _sync_live_positions_and_trades(self, *, force: bool = False, backfill_trades: bool = False) -> None:
        updated = self.balance_manager.sync_positions(force=force)
        if updated is not None:
            status_entries: List[Dict[str, Any]] = []
            for symbol, snapshot in updated.items():
                status_entry = self._format_exchange_status(snapshot)
                if status_entry:
                    status_entries.append(status_entry)
            if not status_entries:
                for ctx in self.contexts:
                    self._clear_local_position(ctx, reason="exchange_flat")
            total_open = sum(float(pos.get("pnl", 0.0) or 0.0) for pos in status_entries)
            self._last_position_snapshot = status_entries
            status_store.set_positions(status_entries)
            status_store.set_open_pnl(total_open)
            self._publish_symbol_summaries(status_entries)
            self._reconcile_position_cache(status_entries)
        self._sync_recent_trades(force=force, backfill=backfill_trades)

    def _format_exchange_status(self, snapshot: ExchangePosition) -> Optional[Dict[str, Any]]:
        ctx = self._ctx_lookup.get(snapshot.symbol)
        if not ctx:
            return None
        self._ensure_ctx_position_alignment(ctx, snapshot)
        metadata = ctx.position.metadata if ctx.position else {}
        sizing_meta = metadata.get("sizing", {}) if metadata else {}
        return {
            "symbol": ctx.symbol,
            "timeframe": ctx.timeframe,
            "side": "LONG" if snapshot.direction == 1 else "SHORT",
            "quantity": snapshot.quantity,
            "entry_price": snapshot.entry_price,
            "mark_price": snapshot.mark_price,
            "pnl": snapshot.pnl,
            "position_side": snapshot.position_side,
            "leverage": snapshot.leverage,
            "sizing_mode": sizing_meta.get("mode", self._config.sizing.mode),
            "stop_loss": ctx.position.stop_loss if ctx.position else None,
            "take_profit": ctx.position.take_profit if ctx.position else None,
            "volatility": sizing_meta.get("volatility"),
            "mta": metadata.get("mta") if metadata else None,
            "signals": metadata.get("external_signals") if metadata else None,
        }

    def _exchange_min_qty(self, symbol: str) -> float:
        filters = self.exchange_info.get_filters(symbol)
        if not filters:
            return 0.0
        return max(filters.min_qty, 0.0)

    def _ensure_ctx_position_alignment(self, ctx: MarketContext, snapshot: ExchangePosition) -> None:
        direction = snapshot.direction
        quantity = snapshot.quantity
        entry_price = snapshot.entry_price or snapshot.mark_price
        mark_price = snapshot.mark_price or entry_price
        timestamp = snapshot.timestamp * 1000
        if ctx.position is None:
            stop_loss, take_profit = self._synthetic_levels(ctx, direction, entry_price)
            ctx.position = PaperPosition(
                symbol=ctx.symbol,
                timeframe=ctx.timeframe,
                direction=direction,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                opened_at=self._format_timestamp(timestamp),
                metadata={"synced": True, "sizing": {"mode": self._config.sizing.mode}},
            )
        else:
            ctx.position.direction = direction
            ctx.position.quantity = quantity
            ctx.position.entry_price = entry_price
            ctx.position.metadata.setdefault("sizing", {}).setdefault("mode", self._config.sizing.mode)
        if ctx.position and ctx.position.metadata.get("synced"):
            stop_loss, take_profit = self._synthetic_levels(ctx, direction, entry_price)
            ctx.position.stop_loss = stop_loss
            ctx.position.take_profit = take_profit

    def _synthetic_levels(self, ctx: MarketContext, direction: int, entry_price: float) -> Tuple[float, float]:
        atr_value = self._estimate_atr(ctx)
        if atr_value <= 0:
            atr_value = max(entry_price * 0.001, 0.1)
        stop_distance = atr_value * max(ctx.params.atr_stop, 0.5)
        target_distance = atr_value * max(ctx.params.atr_target, 0.5)
        if direction == 1:
            return entry_price - stop_distance, entry_price + target_distance
        return entry_price + stop_distance, entry_price - target_distance

    def _estimate_atr(self, ctx: MarketContext) -> float:
        frame = self._latest_frames.get(ctx.symbol)
        if frame is None or frame.empty:
            return 0.0
        snapshot = volatility_snapshot(frame, self._config)
        return float(snapshot.get("atr", 0.0) or 0.0)

    def _clear_local_position(self, ctx: MarketContext, *, reason: str) -> None:
        if not ctx.position:
            return
        ctx.position = None
        logger.info("Cleared local %s %s position (%s)", ctx.symbol, ctx.timeframe, reason)

    def _reconcile_after_reduce_only(self, ctx: MarketContext, direction: int) -> None:
        self.balance_manager.sync_positions(force=True)
        live_qty = self.balance_manager.live_position_quantity(ctx.symbol, direction)
        if live_qty <= 0:
            self._clear_local_position(ctx, reason="exchange_closed")
            return
        if ctx.position:
            ctx.position.quantity = live_qty

    def _force_flatten_positions(self, reason: str) -> None:  # type: ignore[override]
        self._emit_kill_switch_event(reason)
        super()._force_flatten_positions(reason)

    def _sync_recent_trades(self, *, force: bool = False, backfill: bool = False) -> None:
        now = time.time()
        if not force and not backfill and (now - self._last_trade_refresh) < self._trade_refresh_interval:
            return
        self._last_trade_refresh = now
        for ctx in self.contexts:
            try:
                trades = self.client.get_account_trades(symbol=ctx.symbol, limit=50)
            except ExchangeRequestError as exc:
                logger.warning("Account trades fetch failed for %s: %s", ctx.symbol, exc)
                continue
            if not trades:
                continue
            ordered = sorted(trades, key=lambda item: self._int_or_default(item.get("id")))
            last_seen = self._last_trade_ids.get(ctx.symbol, 0)
            new_payloads: List[Dict[str, Any]] = []
            for raw in ordered:
                trade_id = self._int_or_default(raw.get("id"))
                if not backfill and trade_id <= last_seen:
                    continue
                order_id = self._int_or_none(raw.get("orderId"))
                if order_id is not None and order_id in self._synced_order_ids:
                    continue
                pnl_value = self._safe_float(raw.get("realizedPnl"), 0.0)
                if abs(pnl_value) < 1e-9:
                    continue
                if not backfill and raw.get("time") and int(raw["time"]) < self._session_start_ms:
                    continue
                payload = self._format_trade_payload(ctx, raw, pnl_value)
                if not payload:
                    continue
                if order_id is not None:
                    self._synced_order_ids.add(order_id)
                new_payloads.append(payload)
            if new_payloads:
                for payload in new_payloads:
                    pnl_val = float(payload.get("pnl", 0.0) or 0.0)
                    status_store.add_trade(payload)
                    status_store.add_live_trade(payload)
                    stats = self._stats[self._ctx_key(ctx)]
                    stats.record_trade(pnl_val, payload)
                    self._pnl_history.append(pnl_val)
                    trade_epoch = self._timestamp_to_epoch(payload.get("closed_at"))
                    self._risk_engine.register_trade(
                        TradeEvent(pnl=pnl_val, equity=self._estimate_equity(), timestamp=trade_epoch, symbol=ctx.symbol)
                    )
                    if len(self._pnl_history) > 500:
                        self._pnl_history = self._pnl_history[-500:]
                self._last_trade_ids[ctx.symbol] = self._int_or_default(new_payloads[-1].get("exchange_trade_id"))
            elif ordered:
                self._last_trade_ids[ctx.symbol] = max(last_seen, self._int_or_default(ordered[-1].get("id")))

    def _format_trade_payload(
        self,
        ctx: MarketContext,
        trade: Dict[str, Any],
        pnl_value: float,
    ) -> Optional[Dict[str, Any]]:
        price = self._safe_float(trade.get("price"), 0.0)
        quantity = abs(self._safe_float(trade.get("qty"), 0.0))
        if price <= 0 or quantity <= 0:
            return None
        timestamp = self._format_timestamp(trade.get("time"))
        trade_id = self._int_or_default(trade.get("id"))
        return {
            "mode": self.mode_label,
            "symbol": ctx.symbol,
            "timeframe": ctx.timeframe,
            "side": self._infer_trade_direction(trade),
            "quantity": quantity,
            "entry_price": price,
            "exit_price": price,
            "pnl": pnl_value,
            "reason": "live_fill",
            "opened_at": timestamp,
            "closed_at": timestamp,
            "exchange_trade_id": trade_id,
            "order_id": trade.get("orderId"),
        }

    @staticmethod
    def _format_timestamp(raw: Any) -> str:
        try:
            value = float(raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if value > 1_000_000_000_000:
            value /= 1000.0
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))

    @staticmethod
    def _int_or_none(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    @staticmethod
    def _int_or_default(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return default

    def _infer_trade_direction(self, trade: Dict[str, Any]) -> str:
        position_side = (trade.get("positionSide") or "").upper()
        if position_side in {"LONG", "SHORT"}:
            return position_side
        side = (trade.get("side") or "").upper()
        if side == "BUY":
            return "LONG"
        if side == "SELL":
            return "SHORT"
        qty = self._safe_float(trade.get("qty"), 0.0)
        return "LONG" if qty >= 0 else "SHORT"

    def _position_cache_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}::{timeframe or 'NA'}"

    def _snapshot_from_position(self, ctx: MarketContext, position: PaperPosition) -> Dict[str, Any]:
        snapshot = position.as_status(position.entry_price, 0.0)
        snapshot["opened_at"] = position.opened_at
        return snapshot

    def _record_position_open(self, ctx: MarketContext, position: PaperPosition, *, source: str) -> None:
        snapshot = self._snapshot_from_position(ctx, position)
        key = self._position_cache_key(ctx.symbol, ctx.timeframe)
        self._position_cache[key] = snapshot
        self._pending_closures.discard(key)
        self._emit_snapshot_open(snapshot, source=source)

    def _record_position_close(self, ctx: MarketContext, trade_payload: Dict[str, Any]) -> None:
        key = self._position_cache_key(ctx.symbol, ctx.timeframe)
        cached = dict(self._position_cache.get(key) or {})
        if not cached:
            cached = {
                "symbol": ctx.symbol,
                "timeframe": ctx.timeframe,
                "side": trade_payload.get("side"),
                "quantity": trade_payload.get("quantity"),
                "entry_price": trade_payload.get("entry_price"),
                "stop_loss": trade_payload.get("stop_loss"),
                "take_profit": trade_payload.get("take_profit"),
            }
        cached.update(
            {
                "mark_price": trade_payload.get("exit_price"),
                "pnl": trade_payload.get("pnl"),
            }
        )
        self._position_cache[key] = cached
        self._pending_closures.add(key)
        self._emit_snapshot_close(
            cached,
            reason=trade_payload.get("reason", "exit"),
            exit_price=trade_payload.get("exit_price"),
            pnl=trade_payload.get("pnl"),
        )

    def _reconcile_position_cache(self, snapshots: List[Dict[str, Any]]) -> None:
        if not snapshots and not self._position_cache:
            return
        previous_keys = set(self._position_cache.keys())
        current_keys: set[str] = set()
        for snapshot in snapshots:
            symbol = str(snapshot.get("symbol") or "").upper()
            timeframe = str(snapshot.get("timeframe") or "")
            if not symbol:
                continue
            key = self._position_cache_key(symbol, timeframe)
            current_keys.add(key)
            if key not in self._position_cache:
                self._position_cache[key] = snapshot
                self._emit_snapshot_open(snapshot, source="exchange_sync")
            else:
                self._position_cache[key] = snapshot
        removed = previous_keys - current_keys
        for key in removed:
            snapshot = self._position_cache.pop(key, None)
            if snapshot is None:
                continue
            if key in self._pending_closures:
                self._pending_closures.discard(key)
                continue
            self._emit_snapshot_close(snapshot, reason="exchange_sync")

    def _emit_snapshot_open(self, snapshot: Dict[str, Any], *, source: str) -> None:
        payload = self._compact_fields(
            {
                "run_mode": self._config.run_mode,
                "symbol": snapshot.get("symbol"),
                "timeframe": snapshot.get("timeframe"),
                "side": snapshot.get("side"),
                "qty": snapshot.get("quantity"),
                "entry_price": snapshot.get("entry_price"),
                "stop_loss": snapshot.get("stop_loss"),
                "take_profit": snapshot.get("take_profit"),
                "source": source,
                "position": snapshot,
            }
        )
        log_event("POSITION_OPENED", **payload)

    def _emit_snapshot_close(
        self,
        snapshot: Dict[str, Any],
        *,
        reason: str,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
    ) -> None:
        payload = self._compact_fields(
            {
                "run_mode": self._config.run_mode,
                "symbol": snapshot.get("symbol"),
                "timeframe": snapshot.get("timeframe"),
                "side": snapshot.get("side"),
                "qty": snapshot.get("quantity"),
                "entry_price": snapshot.get("entry_price"),
                "exit_price": exit_price or snapshot.get("mark_price"),
                "pnl": pnl or snapshot.get("pnl"),
                "reason": reason,
                "position": snapshot,
            }
        )
        log_event("POSITION_CLOSED", **payload)

    def _emit_equity_event(self, snapshot: BalanceSnapshot) -> None:
        open_pnl = sum(float(pos.get("pnl", 0.0) or 0.0) for pos in self._last_position_snapshot)
        payload = self._compact_fields(
            {
                "run_mode": self._config.run_mode,
                "equity": self._estimate_equity(),
                "balance": snapshot.available,
                "total_balance": snapshot.total,
                "unrealized_pnl": open_pnl,
                "open_positions": len(self._last_position_snapshot),
                "account_asset": self._account_asset,
            }
        )
        log_event("EQUITY_SNAPSHOT", **payload)

    def _emit_kill_switch_event(self, reason: Optional[str]) -> None:
        state = self._risk_engine.current_state()
        key = (reason or "unspecified", state.last_triggered_at or 0.0)
        if self._kill_switch_event_key == key:
            return
        self._kill_switch_event_key = key
        payload = self._compact_fields(
            {
                "run_mode": self._config.run_mode,
                "reason": reason or state.last_trigger_reason or "unspecified",
                "equity": state.equity,
                "pnl_today": state.pnl_today,
                "loss_abs": state.loss_abs,
                "loss_pct": state.loss_pct,
                "loss_streak": state.loss_streak,
                "max_consecutive_losses": state.max_consecutive_losses,
                "reference_equity": state.reference_equity,
                "trading_mode": state.trading_mode.value,
            }
        )
        log_event("KILL_SWITCH_TRIGGERED", **payload)

    @staticmethod
    def _compact_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in data.items() if value is not None}
__all__ = ["LiveTrader"]
