"""Live trading runner that extends the multi-symbol loop with testnet orders."""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bot.core.config import BotConfig
from bot.exchange_info import ExchangeInfoManager
from bot.execution.exchange_client import ExchangeClient, ExchangeRequestError
from bot.learning import TradeLearningStore
from bot.live_logging import LiveTradeLogger, OrderAuditLogger
from bot.execution.runners import MarketContext, MultiSymbolRunnerBase, PaperPosition
from bot.risk import RiskEngine, volatility_snapshot
from bot.signals.strategies import StrategyParameters, StrategySignal
from bot.status import status_store
from bot.utils.logger import get_logger

logger = get_logger(__name__)


class LiveTrader(MultiSymbolRunnerBase):
    _TRANSIENT_ERRORS = {-1003, -1015, -1021}
    _EXPECTED_WARNINGS = {-2019, -2021, -2026}

    def __init__(
        self,
        markets: Sequence[Tuple[str, str, StrategyParameters]],
        exchange_info: ExchangeInfoManager,
        *,
        cfg: BotConfig,
        client: ExchangeClient,
        portfolio_meta: Optional[Dict[str, Any]] = None,
        learning_store: Optional[TradeLearningStore] = None,
    ) -> None:
        self._config = cfg
        self.client = client
        self.order_logger = OrderAuditLogger()
        self.trade_logger = LiveTradeLogger()
        self.learning_store = learning_store or TradeLearningStore()
        self._applied_learning: Dict[str, Dict[str, Any]] = {}
        self._account_asset = cfg.runtime.demo_account_asset.upper()
        self._balance_refresh_interval = cfg.runtime.balance_refresh_seconds
        self._last_balance_refresh = 0.0
        self._last_position_refresh = 0.0
        self._last_trade_refresh = 0.0
        self._last_position_snapshot: List[Dict[str, Any]] = []
        self._last_trade_ids: Dict[str, int] = {}
        self._synced_order_ids: set[int] = set()
        self._session_start_ms = int(time.time() * 1000)
        self._exchange_positions: Dict[str, Dict[str, Any]] = {}
        self._latest_frames: Dict[str, Any] = {}
        self._position_log_cache: Dict[str, float] = {}
        self._risk_engine = RiskEngine(cfg)
        self._last_learning_update: Dict[str, float] = {}
        self.balance = cfg.runtime.paper_account_balance
        self.available_balance = cfg.runtime.paper_account_balance
        self.total_balance = cfg.runtime.paper_account_balance
        initial_balance = self._refresh_account_balance(force=True)
        super().__init__(
            markets,
            exchange_info,
            cfg,
            mode_label="live",
            portfolio_meta=portfolio_meta,
            initial_balance=initial_balance,
            status_via_exchange=True,
            risk_engine=self._risk_engine,
        )
        self._ctx_lookup: Dict[str, MarketContext] = {ctx.symbol: ctx for ctx in self.contexts}
        self._position_refresh_interval = max(3.0, min(float(self.poll_interval), 15.0))
        self._trade_refresh_interval = max(5.0, min(float(self.poll_interval) * 2.0, 30.0))

    async def _before_loop(self) -> None:
        for ctx in self.contexts:
            self._ensure_leverage(ctx.symbol)
        self._sync_live_positions_and_trades(force=True, backfill_trades=True)
        await super()._before_loop()

    async def _before_step(self) -> None:
        self._refresh_account_balance()
        self._apply_learned_parameters()
    def _step_all(self) -> None:
        self._sync_live_positions_and_trades()
        super()._step_all()
        self._sync_live_positions_and_trades()

    def _fetch_frame(self, ctx: MarketContext, lookback: int):  # type: ignore[override]
        frame = super()._fetch_frame(ctx, lookback)
        if frame is not None and not frame.empty:
            self._latest_frames[ctx.symbol] = frame
        return frame

    def _balance_for_risk(self) -> float:
        return self._refresh_account_balance()

    def _ensure_leverage(self, symbol: str) -> None:
        try:
            self.client.change_leverage(symbol=symbol, leverage=int(self._config.risk.leverage))
        except ExchangeRequestError as exc:
            logger.warning("Unable to set leverage for %s: %s", symbol, exc)

    def _refresh_account_balance(self, force: bool = False) -> float:
        now = time.time()
        if not force and (now - self._last_balance_refresh) < self._balance_refresh_interval:
            return self.balance
        total_value = self.total_balance
        available_value = self.available_balance
        try:
            balances = self.client.get_balance()
        except ExchangeRequestError as exc:
            logger.warning("Balance refresh failed: %s", exc)
            return self.balance
        for entry in balances or []:
            if entry.get("asset") == self._account_asset:
                total_value = self._safe_float(entry.get("balance"), total_value)
                available_raw = (
                    entry.get("availableBalance")
                    or entry.get("withdrawAvailable")
                    or entry.get("crossWalletBalance")
                    or entry.get("balance")
                )
                available_value = self._safe_float(available_raw, available_value)
                break
        self.total_balance = total_value
        self.available_balance = available_value
        self.balance = available_value
        self._last_balance_refresh = now
        self._risk_engine.on_balance_refresh(self.available_balance)
        status_store.update_balance(available_value)
        return available_value

    def _submit_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        *,
        reduce_only: bool = False,
        price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        filters = self.exchange_info.get_filters(symbol)
        attempt_qty = max(quantity, 0.0)
        if reduce_only:
            attempt_qty = self._resolve_reduce_only_quantity(symbol, side, attempt_qty, price, filters)
            if attempt_qty <= 0:
                return None
        elif filters:
            attempt_qty = filters.adjust_quantity(attempt_qty)
        if attempt_qty <= 0:
            self._log_order_skip(symbol, side, "non_positive_qty", reduce_only)
            return None
        for attempt in range(2):
            payload = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": self._format_quantity(attempt_qty),
            }
            if reduce_only:
                payload["reduceOnly"] = "true"
            self.order_logger.log({"event": "request", **payload})
            try:
                response = self.client.place_order(**payload)
            except ExchangeRequestError as exc:
                self.order_logger.log({"event": "error", "symbol": symbol, "message": str(exc)})
                code = exc.code
                self._log_client_order_error(symbol, side, attempt_qty, code, exc)
                if code == -2019 and not reduce_only and attempt == 0:
                    attempt_qty = self._reduce_quantity_after_margin_error(symbol, attempt_qty, filters)
                    if attempt_qty <= 0:
                        self._log_order_skip(symbol, side, "margin", reduce_only)
                        return None
                    continue
                if code in self._TRANSIENT_ERRORS and attempt == 0:
                    time.sleep(0.5)
                    continue
                return None
            except Exception as exc:  # pragma: no cover - defensive
                self.order_logger.log({"event": "error", "symbol": symbol, "message": str(exc)})
                logger.error("%s order failed for %s: %s", side, symbol, exc)
                return None
            self.order_logger.log({"event": "response", **response})
            return response
        return None

    def _on_position_open_request(
        self,
        ctx: MarketContext,
        position: PaperPosition,
        latest_row: Any,
        signal: StrategySignal,
    ) -> bool:
        del latest_row, signal
        side = "BUY" if position.direction == 1 else "SELL"
        response = self._submit_market_order(ctx.symbol, side, position.quantity)
        if not response:
            return False
        avg_price = self._parse_price(response) or position.entry_price
        position.entry_price = avg_price
        executed_qty = self._parse_quantity(response)
        if executed_qty and executed_qty > 0:
            position.quantity = executed_qty
        position.metadata.setdefault("indicators", {})
        position.metadata["entry_order_id"] = response.get("orderId")
        position.metadata["opened_at"] = self._timestamp_from_response(response)
        return True

    def _on_position_close_request(
        self,
        ctx: MarketContext,
        position: PaperPosition,
        exit_price: float,
        exit_reason: str,
    ) -> bool:
        del exit_price, exit_reason
        side = "SELL" if position.direction == 1 else "BUY"
        self._ensure_position_snapshot(force=True)
        live_qty = self._live_position_quantity(ctx.symbol, position.direction)
        if live_qty <= 0:
            self._clear_local_position(ctx, reason="exchange_flat")
            logger.info("Skipping reduce-only close for %s %s: no exchange position", ctx.symbol, ctx.timeframe)
            return False
        if position.quantity > live_qty:
            position.quantity = live_qty
        response = self._submit_market_order(ctx.symbol, side, position.quantity, reduce_only=True, price=exit_price)
        if not response:
            return False
        position.metadata["exit_order_id"] = response.get("orderId")
        position.metadata["closed_at"] = self._timestamp_from_response(response)
        self._reconcile_after_reduce_only(ctx, position.direction)
        return True

    def _after_trade_closed(self, ctx: MarketContext, trade_payload: Dict[str, Any]) -> None:
        self.trade_logger.log(trade_payload)
        self.learning_store.record_trade(trade_payload, ctx.params)
        exit_order_id = self._int_or_none(trade_payload.get("exit_order_id"))
        if exit_order_id is not None:
            self._synced_order_ids.add(exit_order_id)

    def _apply_learned_parameters(self) -> None:
        for ctx in self.contexts:
            overrides = self.learning_store.best_params(ctx.symbol, ctx.timeframe)
            if not overrides:
                continue
            key = self._ctx_key(ctx)
            trade_samples = self.learning_store.trade_count(ctx.symbol, ctx.timeframe)
            reinforcement_cfg = self._config.reinforcement
            if trade_samples < reinforcement_cfg.min_trades_before_update:
                continue
            now = time.time()
            last_update = self._last_learning_update.get(key, 0.0)
            if (now - last_update) < reinforcement_cfg.update_cooldown_seconds:
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

    def _merge_parameters(self, current: StrategyParameters, overrides: Dict[str, Any]) -> StrategyParameters:
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
        available = max(self.available_balance, 0.0)
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
        if reason == "margin" and not self._risk_engine.should_log_margin_block(ctx.symbol, self.available_balance):
            return
        super()._log_sizing_skip(ctx, reason)

    def _equity_for_risk(self) -> float:
        return self._estimate_equity()

    def _available_margin_for_risk(self) -> float:
        return max(self.available_balance, 0.0)

    def _exposure_for_entry(self, ctx: MarketContext, price: float) -> Tuple[float, float, int, bool]:
        state = self._risk_engine.compute_exposure(self._exchange_positions)
        symbol_total = state.per_symbol.get(ctx.symbol, 0.0)
        active_symbols = sum(1 for value in state.per_symbol.values() if value > 0)
        return symbol_total, state.total, active_symbols, symbol_total > 0

    def _estimate_equity(self) -> float:
        open_pnl = sum(float(pos.get("pnl", 0.0) or 0.0) for pos in self._last_position_snapshot)
        baseline = max(self.total_balance, self.available_balance)
        return max(baseline + open_pnl, 0.0)

    @staticmethod
    def _safe_float(value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
            if math.isnan(parsed) or math.isinf(parsed):
                return fallback
            return parsed
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return fallback

    @staticmethod
    def _format_quantity(quantity: float) -> str:
        return f"{max(quantity, 0.0):.8f}"

    @staticmethod
    def _parse_quantity(response: Dict[str, Any]) -> Optional[float]:
        for key in ("executedQty", "cumQty", "origQty", "quantity"):
            raw = response.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            if value > 0:
                return value
        return None

    def _reduce_quantity_after_margin_error(self, symbol: str, quantity: float, filters: Optional[Any]) -> float:
        del symbol
        reduced = quantity * 0.5
        if filters:
            reduced = filters.adjust_quantity(reduced)
            if reduced < filters.min_qty:
                return 0.0
        return reduced

    def _log_order_skip(self, symbol: str, side: str, reason: str, reduce_only: bool) -> None:
        mode = "reduce-only" if reduce_only else "market"
        logger.info("[%s][%s] Skipping %s order: %s", symbol, side, mode, reason)

    def _log_reduce_only_skip(self, symbol: str, side: str, reason: str) -> None:
        cache_key = f"{symbol}:{side}:{reason}"
        now = time.time()
        if now - self._position_log_cache.get(cache_key, 0.0) < 5.0:
            return
        self._position_log_cache[cache_key] = now
        logger.info("[%s][%s] Reduce-only skipped (%s)", symbol, side, reason)

    def _log_client_order_error(self, symbol: str, side: str, quantity: float, code: Any, exc: Exception) -> None:
        severity = logging.INFO if code in self._EXPECTED_WARNINGS else logging.WARNING
        logger.log(
            severity,
            "[%s][%s] Order rejected qty=%.6f code=%s msg=%s",
            symbol,
            side,
            quantity,
            code,
            exc,
        )

    def _resolve_reduce_only_quantity(
        self,
        symbol: str,
        side: str,
        requested: float,
        price: Optional[float],
        filters: Optional[Any],
    ) -> float:
        direction = 1 if side == "SELL" else -1
        self._ensure_position_snapshot(force=True)
        live_qty = self._live_position_quantity(symbol, direction)
        if live_qty <= 0:
            self._log_reduce_only_skip(symbol, side, "exchange_flat")
            return 0.0
        qty = min(requested, live_qty)
        if filters:
            qty = filters.adjust_quantity(qty)
            mark_price = (
                self._exchange_positions.get(symbol, {}).get("mark_price")
                or self._exchange_positions.get(symbol, {}).get("entry_price")
                or price
                or 0.0
            )
            if qty < filters.min_qty or (mark_price * qty) < filters.min_notional:
                self._log_reduce_only_skip(symbol, side, "min_notional")
                return 0.0
        return qty

    def _sync_live_positions_and_trades(self, *, force: bool = False, backfill_trades: bool = False) -> None:
        snapshot = self._refresh_live_positions(force=force)
        if snapshot is not None:
            total_open = sum(float(pos.get("pnl", 0.0) or 0.0) for pos in snapshot)
            status_store.set_positions(snapshot)
            status_store.set_open_pnl(total_open)
            self._publish_symbol_summaries(snapshot)
        self._sync_recent_trades(force=force, backfill=backfill_trades)

    def _refresh_live_positions(self, force: bool = False) -> Optional[List[Dict[str, Any]]]:
        now = time.time()
        if not force and (now - self._last_position_refresh) < self._position_refresh_interval:
            return None
        try:
            payload = self.client.get_position_risk()
        except ExchangeRequestError as exc:
            logger.warning("Position risk refresh failed: %s", exc)
            return None
        positions: List[Dict[str, Any]] = []
        active_symbols: set[str] = set()
        for entry in payload or []:
            status_entry = self._process_position_entry(entry)
            if status_entry:
                positions.append(status_entry)
                active_symbols.add(status_entry["symbol"])
        if not active_symbols:
            for ctx in self.contexts:
                self._exchange_positions.pop(ctx.symbol, None)
                self._clear_local_position(ctx, reason="exchange_flat")
        self._last_position_refresh = now
        self._last_position_snapshot = positions
        return positions

    def _process_position_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = str(entry.get("symbol") or "").upper()
        ctx = self._ctx_lookup.get(symbol)
        if not ctx:
            return None
        position_side = (entry.get("positionSide") or "BOTH").upper()
        if position_side not in {"BOTH", "LONG", "SHORT"}:
            return None
        raw_qty = self._safe_float(entry.get("positionAmt"), 0.0)
        min_qty = self._exchange_min_qty(symbol)
        if abs(raw_qty) < max(min_qty, 1e-6):
            self._exchange_positions.pop(symbol, None)
            self._clear_local_position(ctx, reason="exchange_flat")
            return None
        direction = 1 if raw_qty > 0 else -1
        quantity = abs(raw_qty)
        entry_price = self._safe_float(entry.get("entryPrice"), 0.0)
        if entry_price <= 0:
            entry_price = self._safe_float(entry.get("markPrice"), 0.0)
        mark_price = self._safe_float(entry.get("markPrice"), entry_price)
        pnl = self._safe_float(
            entry.get("unRealizedProfit"),
            (mark_price - entry_price) * direction * quantity,
        )
        leverage = self._safe_float(entry.get("leverage"), self._config.risk.leverage)
        self._exchange_positions[symbol] = {
            "quantity": quantity,
            "direction": direction,
            "min_qty": max(min_qty, 1e-6),
            "entry_price": entry_price,
            "mark_price": mark_price,
            "notional": quantity * mark_price,
            "timestamp": time.time(),
        }
        self._ensure_ctx_position_alignment(ctx, entry, direction, quantity, entry_price)
        metadata = ctx.position.metadata if ctx.position else {}
        sizing_meta = metadata.get("sizing", {}) if metadata else {}
        return {
            "symbol": ctx.symbol,
            "timeframe": ctx.timeframe,
            "side": "LONG" if direction == 1 else "SHORT",
            "quantity": quantity,
            "entry_price": entry_price,
            "mark_price": mark_price,
            "pnl": pnl,
            "position_side": position_side,
            "leverage": leverage,
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

    def _ensure_ctx_position_alignment(
        self,
        ctx: MarketContext,
        entry: Dict[str, Any],
        direction: int,
        quantity: float,
        entry_price: float,
    ) -> None:
        timestamp = entry.get("updateTime") or entry.get("time") or time.time() * 1000
        mark_price = self._safe_float(entry.get("markPrice"), entry_price)
        if entry_price <= 0:
            entry_price = mark_price
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
        if ctx.position.metadata.get("synced"):
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
        snapshot = volatility_snapshot(frame)
        return float(snapshot.get("atr", 0.0) or 0.0)

    def _clear_local_position(self, ctx: MarketContext, *, reason: str) -> None:
        if not ctx.position:
            return
        ctx.position = None
        logger.info("Cleared local %s %s position (%s)", ctx.symbol, ctx.timeframe, reason)

    def _reconcile_after_reduce_only(self, ctx: MarketContext, direction: int) -> None:
        self._sync_live_positions_and_trades(force=True)
        live_qty = self._live_position_quantity(ctx.symbol, direction)
        if live_qty <= 0:
            self._clear_local_position(ctx, reason="exchange_closed")
            return
        if ctx.position:
            ctx.position.quantity = live_qty

    def _ensure_position_snapshot(self, *, force: bool = False) -> None:
        self._refresh_live_positions(force=force)

    def _live_position_quantity(self, symbol: str, direction: int) -> float:
        info = self._exchange_positions.get(symbol)
        if info and info.get("direction") == direction and info.get("quantity", 0.0) > info.get("min_qty", 0.0):
            return float(info.get("quantity", 0.0))
        return 0.0

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
                    self._risk_engine.register_trade(pnl_val, equity=self._estimate_equity(), timestamp=trade_epoch)
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
__all__ = ["LiveTrader"]
