from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from bot.core.config import BotConfig
from bot.exchange_info import ExchangeInfoManager, SymbolFilters
from bot.execution.exchange_client import ExchangeClient, ExchangeRequestError
from bot.risk.engine import RiskEngine
from bot.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class BalanceSnapshot:
    total: float
    available: float
    timestamp: float


@dataclass(slots=True)
class ExchangePosition:
    symbol: str
    direction: int
    quantity: float
    entry_price: float
    mark_price: float
    leverage: float
    position_side: str
    pnl: float
    timestamp: float

    @property
    def notional(self) -> float:
        return self.quantity * self.mark_price


class BalanceManager:
    """Maintains cached account balances and live position state."""

    def __init__(
        self,
        cfg: BotConfig,
        client: ExchangeClient,
        *,
        risk_engine: RiskEngine,
        exchange_info: ExchangeInfoManager,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._cfg = cfg
        self._client = client
        self._risk_engine = risk_engine
        self._exchange_info = exchange_info
        self._clock = clock or time.time
        self._account_asset = cfg.runtime.demo_account_asset.upper()
        self._balance_refresh_interval = max(cfg.runtime.balance_refresh_seconds, 5)
        self._position_refresh_interval = max(float(cfg.runtime.poll_interval_seconds), 3.0)
        now = self._clock()
        starting_balance = cfg.runtime.paper_account_balance
        self._snapshot = BalanceSnapshot(total=starting_balance, available=starting_balance, timestamp=now)
        self._positions: Dict[str, ExchangePosition] = {}
        self._raw_positions: Dict[str, Dict[str, float]] = {}
        self._last_balance_refresh = 0.0
        self._last_position_refresh = 0.0

    @property
    def available_balance(self) -> float:
        return self._snapshot.available

    @property
    def total_balance(self) -> float:
        return self._snapshot.total

    @property
    def raw_positions(self) -> Dict[str, Dict[str, float]]:
        return self._raw_positions

    def exposure_payload(self) -> Dict[str, Dict[str, float]]:
        return dict(self._raw_positions)

    def refresh_account_balance(self, *, force: bool = False) -> BalanceSnapshot:
        now = self._clock()
        if not force and (now - self._last_balance_refresh) < self._balance_refresh_interval:
            return self._snapshot
        total_value = self._snapshot.total
        available_value = self._snapshot.available
        try:
            balances = self._client.get_balance()
        except ExchangeRequestError as exc:
            logger.warning("Balance refresh failed: %s", exc)
            return self._snapshot
        for entry in balances or []:
            asset = str(entry.get("asset") or "").upper()
            if asset != self._account_asset:
                continue
            total_value = self._safe_float(entry.get("balance"), total_value)
            available_raw = (
                entry.get("availableBalance")
                or entry.get("withdrawAvailable")
                or entry.get("crossWalletBalance")
                or entry.get("balance")
            )
            available_value = self._safe_float(available_raw, available_value)
            break
        self._snapshot = BalanceSnapshot(total=total_value, available=available_value, timestamp=now)
        self._last_balance_refresh = now
        self._risk_engine.on_balance_refresh(self._snapshot.available)
        return self._snapshot

    def sync_positions(self, *, force: bool = False) -> Optional[Dict[str, ExchangePosition]]:
        now = self._clock()
        if not force and (now - self._last_position_refresh) < self._position_refresh_interval:
            return None
        try:
            payload = self._client.get_position_risk()
        except ExchangeRequestError as exc:
            logger.warning("Position risk refresh failed: %s", exc)
            return None
        updated: Dict[str, ExchangePosition] = {}
        raw_payload: Dict[str, Dict[str, float]] = {}
        for entry in payload or []:
            snapshot = self._parse_position(entry)
            if not snapshot:
                continue
            updated[snapshot.symbol] = snapshot
            raw_payload[snapshot.symbol] = {
                "quantity": snapshot.quantity,
                "direction": snapshot.direction,
                "entry_price": snapshot.entry_price,
                "mark_price": snapshot.mark_price,
            }
        self._positions = updated
        self._raw_positions = raw_payload
        self._last_position_refresh = now
        return updated

    def positions_snapshot(self) -> Dict[str, ExchangePosition]:
        return dict(self._positions)

    def estimate_equity(self) -> float:
        base = max(self._snapshot.total, self._snapshot.available)
        open_pnl = sum(pos.pnl for pos in self._positions.values())
        return max(base + open_pnl, 0.0)

    def live_position_quantity(self, symbol: str, direction: int) -> float:
        position = self._positions.get(symbol.upper())
        if not position or position.direction != direction:
            return 0.0
        return position.quantity

    def resolve_reduce_only_quantity(
        self,
        symbol: str,
        side: str,
        requested: float,
        price: float | None,
        filters: SymbolFilters | None,
    ) -> float:
        direction = 1 if side.upper() == "SELL" else -1
        live_qty = self.live_position_quantity(symbol, direction)
        if live_qty <= 0:
            return 0.0
        qty = min(requested, live_qty)
        if filters:
            qty = filters.adjust_quantity(qty)
            existing = self._positions.get(symbol.upper())
            mark_price = existing.mark_price if existing else (price or 0.0)
            if qty < filters.min_qty or (mark_price * qty) < filters.min_notional:
                return 0.0
        return qty

    @staticmethod
    def _safe_float(value: Any, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        if parsed != parsed or parsed in {float("inf"), float("-inf")}:
            return fallback
        return parsed

    def _parse_position(self, entry: Dict[str, Any]) -> Optional[ExchangePosition]:
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            return None
        filters = self._exchange_info.get_filters(symbol)
        min_qty = filters.min_qty if filters else 0.0
        raw_qty = self._safe_float(entry.get("positionAmt"), 0.0)
        if abs(raw_qty) < max(min_qty, 1e-6):
            return None
        direction = 1 if raw_qty > 0 else -1
        quantity = abs(raw_qty)
        entry_price = self._safe_float(entry.get("entryPrice"), 0.0)
        mark_price = self._safe_float(entry.get("markPrice"), entry_price)
        if entry_price <= 0:
            entry_price = mark_price
        pnl = self._safe_float(entry.get("unRealizedProfit"), (mark_price - entry_price) * direction * quantity)
        leverage = self._safe_float(entry.get("leverage"), self._cfg.risk.leverage)
        position_side = str(entry.get("positionSide") or "BOTH").upper()
        return ExchangePosition(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            mark_price=mark_price,
            leverage=leverage,
            position_side=position_side,
            pnl=pnl,
            timestamp=self._clock(),
        )


__all__ = [
    "BalanceManager",
    "BalanceSnapshot",
    "ExchangePosition",
]
