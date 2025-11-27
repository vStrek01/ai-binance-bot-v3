from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from core.models import MarketState, OrderRequest, RiskConfig
from core.risk_shared import DailyRiskState, apply_notional_limits, evaluate_kill_switch, stop_based_position_size
from exchange.symbols import SymbolInfo, SymbolResolver
from infra.alerts import send_alert
from infra.logging import logger, log_event
from infra.state_store import StateStore


class RiskManager:
    def __init__(
        self,
        config: RiskConfig,
        *,
        symbol_resolver: SymbolResolver | None = None,
        relaxed_drawdown: bool = False,
        safety_limits: Any | None = None,
        state_store: StateStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.symbol_resolver = symbol_resolver
        self.relaxed_drawdown = relaxed_drawdown
        self._last_reset: Optional[datetime] = None
        self._daily_state: Optional[DailyRiskState] = None
        self.kill_switch_active = False
        self.kill_switch_reason: Optional[str] = None
        self._legacy_safety_limits = safety_limits  # kept for compatibility; dedicated risk controls supersede it
        provided_fields = getattr(config, "model_fields_set", set())
        self._symbol_cap_usd: float | None = (
            config.max_symbol_notional_usd if "max_symbol_notional_usd" in provided_fields else None
        )
        self._total_cap_usd: float | None = (
            config.max_total_notional_usd if "max_total_notional_usd" in provided_fields else None
        )
        self.state_store = state_store
        self.run_id = run_id
        self._hydrate_from_store()

    @property
    def daily_start_equity(self) -> Optional[float]:
        return self._daily_state.starting_equity if self._daily_state else None

    @daily_start_equity.setter
    def daily_start_equity(self, value: float) -> None:  # pragma: no cover - compatibility shim
        if self._daily_state:
            self._daily_state.starting_equity = value
        else:
            self._daily_state = DailyRiskState(session_date=date.today(), starting_equity=value)

    def enable_backtest_mode(self) -> None:
        self.relaxed_drawdown = True

    def reset_day_if_needed(self, equity: float) -> None:
        today = date.today()
        if self._daily_state is None or self._daily_state.session_date != today:
            self._daily_state = DailyRiskState(session_date=today, starting_equity=equity)
            self.kill_switch_active = False
            self._last_reset = datetime.utcnow()
            logger.info("Daily risk state initialized", extra={"equity": equity, "date": today.isoformat()})
            self._persist_state()

    def validate(self, order: OrderRequest, state: MarketState) -> Optional[OrderRequest]:
        self.reset_day_if_needed(state.equity)
        self._update_kill_switch(equity=state.equity)
        if self.kill_switch_active:
            logger.warning("Kill switch active; blocking order", extra={"symbol": order.symbol})
            self._log_order_rejected(order, "kill_switch_active", equity=state.equity)
            return None
        if len(state.open_positions) >= self.config.max_open_positions:
            logger.info("Max open positions reached", extra={"symbol": order.symbol})
            self._log_order_rejected(order, "max_open_positions", open_positions=len(state.open_positions))
            return None
        if order.leverage > self.config.max_leverage:
            logger.warning("Leverage too high", extra={"requested": order.leverage})
            self._log_order_rejected(order, "leverage_exceeded", leverage=order.leverage)
            return None
        if order.stop_loss is None:
            logger.warning("Stop loss required for sizing", extra={"symbol": order.symbol})
            self._log_order_rejected(order, "missing_stop_loss")
            return None

        symbol_info = self._resolve_symbol(order.symbol)
        reference_price = order.price or (state.candles[-1].close if state.candles else None)
        if reference_price is None:
            logger.warning("Cannot size order; missing price context", extra={"symbol": order.symbol})
            self._log_order_rejected(order, "missing_price_context")
            return None
        qty = self._size_position(state.equity, reference_price, order.stop_loss, symbol_info)
        if qty <= 0:
            self._log_order_rejected(order, "size_zero", equity=state.equity)
            return None

        qty = self._apply_notional_limits(qty, reference_price, order.symbol, state, symbol_info)
        if qty <= 0:
            self._log_order_rejected(order, "notional_limit", price=reference_price)
            return None

        rounded_price = symbol_info.round_price(reference_price) if symbol_info and reference_price else reference_price
        safe_order = order.model_copy(update={"quantity": qty, "price": rounded_price})
        logger.info(
            "Order approved by risk",
            extra={"symbol": safe_order.symbol, "qty": safe_order.quantity, "price": safe_order.price},
        )
        return safe_order

    def record_fill(
        self,
        pnl: float,
        *,
        commission_pct_of_equity: float = 0.0,
        is_closed_trade: bool = True,
    ) -> None:
        if self._daily_state is None:
            return
        if is_closed_trade:
            self._daily_state.realized_pnl += pnl
            if pnl < 0:
                self._daily_state.consecutive_losses += 1
            elif pnl > 0:
                self._daily_state.consecutive_losses = 0
        self._daily_state.commission_paid_pct += commission_pct_of_equity
        self._update_kill_switch()
        self._persist_state()

    def record_trade_entry(self) -> None:
        if self._daily_state is None:
            return
        self._daily_state.trades_count += 1
        self._update_kill_switch()
        self._persist_state()

    def _size_position(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        symbol_info: SymbolInfo | None,
    ) -> float:
        qty = stop_based_position_size(
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            max_risk_per_trade_pct=self.config.max_risk_per_trade_pct,
            symbol_info=symbol_info,
        )
        if qty <= 0:
            logger.warning("Invalid stop distance", extra={"entry": entry_price, "stop": stop_price})
        return qty

    def _apply_notional_limits(
        self,
        qty: float,
        price: float,
        symbol: str,
        state: MarketState,
        symbol_info: SymbolInfo | None,
    ) -> float:
        symbol_open_notional = sum(
            abs(pos.quantity * pos.entry_price)
            for pos in state.open_positions
            if pos.symbol == symbol and pos.is_open()
        )
        total_notional = sum(abs(pos.quantity * pos.entry_price) for pos in state.open_positions if pos.is_open())
        sized = apply_notional_limits(
            qty,
            price=price,
            min_order_notional=self.config.min_order_notional_usd,
            symbol_info=symbol_info,
            symbol_cap_usd=self._symbol_cap_usd,
            total_cap_usd=self._total_cap_usd,
            symbol_open_notional=symbol_open_notional,
            total_open_notional=total_notional,
        )
        if sized <= 0:
            logger.info(
                "Order below caps",
                extra={"symbol": symbol, "notional": qty * price, "min_notional": self.config.min_order_notional_usd},
            )
        return sized

    def _update_kill_switch(self, equity: Optional[float] = None) -> None:
        reasons = evaluate_kill_switch(
            self.config,
            self._daily_state,
            equity=equity,
            relaxed_drawdown=self.relaxed_drawdown,
        )
        if reasons:
            self._activate_kill_switch(reason="|".join(reasons))
        else:
            self._persist_state()

    def _activate_kill_switch(self, reason: str) -> None:
        if self.kill_switch_active:
            return
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        payload = {
            "reason": reason,
            "drawdown_pct": self._daily_state.drawdown_pct() if self._daily_state else None,
            "trades": self._daily_state.trades_count if self._daily_state else None,
            "consecutive_losses": self._daily_state.consecutive_losses if self._daily_state else None,
        }
        log_event("risk_kill_switch_triggered", **payload)
        log_event("KILL_SWITCH_TRIGGERED", **payload)
        logger.error("Kill switch activated", extra={"reason": reason})
        send_alert(
            "KILL_SWITCH_TRIGGERED",
            severity="critical",
            message="Risk kill switch activated",
            reason=reason,
            drawdown_pct=payload.get("drawdown_pct"),
            trades=payload.get("trades"),
            consecutive_losses=payload.get("consecutive_losses"),
        )
        self._persist_state()

    def _resolve_symbol(self, symbol: str) -> SymbolInfo | None:
        if not self.symbol_resolver:
            return None
        try:
            return self.symbol_resolver.get(symbol)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Symbol resolver missing data", extra={"symbol": symbol, "error": str(exc)})
            return None

    def _log_order_rejected(self, order: OrderRequest, reason: str, **fields: Any) -> None:
        payload = {
            "symbol": order.symbol,
            "side": getattr(order.side, "value", str(order.side)),
            "reason": reason,
            **{k: v for k, v in fields.items() if v is not None},
        }
        log_event("order_rejected_by_risk", **payload)

    # ------------------------------------------------------------------

    def _hydrate_from_store(self) -> None:
        if not self.state_store:
            return
        snapshot = self.state_store.load()
        payload = snapshot.get("risk") if isinstance(snapshot, dict) else None
        if not isinstance(payload, dict):
            return
        session_str = payload.get("session_date")
        try:
            session_date = date.fromisoformat(session_str) if session_str else date.today()
        except ValueError:
            session_date = date.today()
        self._daily_state = DailyRiskState(
            session_date=session_date,
            starting_equity=float(payload.get("starting_equity", 0.0) or 0.0),
            realized_pnl=float(payload.get("realized_pnl", 0.0) or 0.0),
            trades_count=int(payload.get("trades_count", 0) or 0),
            commission_paid_pct=float(payload.get("commission_paid_pct", 0.0) or 0.0),
            consecutive_losses=int(payload.get("consecutive_losses", 0) or 0),
        )
        self.kill_switch_active = bool(payload.get("kill_switch_active", False))
        self.kill_switch_reason = payload.get("kill_switch_reason")

    def _persist_state(self) -> None:
        if not (self.state_store and self._daily_state):
            return
        try:
            payload = {
                "session_date": self._daily_state.session_date.isoformat(),
                "starting_equity": self._daily_state.starting_equity,
                "realized_pnl": self._daily_state.realized_pnl,
                "trades_count": self._daily_state.trades_count,
                "commission_paid_pct": self._daily_state.commission_paid_pct,
                "consecutive_losses": self._daily_state.consecutive_losses,
                "kill_switch_active": self.kill_switch_active,
                "kill_switch_reason": self.kill_switch_reason,
                "updated_at": datetime.utcnow().isoformat(),
                "run_id": self.run_id,
            }
            self.state_store.merge(risk=payload)
            log_event("risk_state_updated", **payload)
        except Exception as exc:  # noqa: BLE001 - persistence must not break trading path
            logger.warning("Failed to persist risk state", extra={"error": str(exc)})
