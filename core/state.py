from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from core.models import OrderFill, Position, Side
from infra.logging import logger, log_event
from infra.state_store import StateStore

MAX_POSITION_PCT = 0.2


@dataclass
class PositionManager:
    positions: Dict[str, Position] = field(default_factory=dict)
    equity: float = 10_000.0
    fees_paid: float = 0.0
    consecutive_losses: int = 0
    run_mode: str = "backtest"
    state_store: StateStore | None = field(default=None, repr=False)
    run_id: str | None = None

    def get_open_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.is_open()]

    def symbol_notional(self, symbol: str) -> float:
        position = self.positions.get(symbol)
        if position and position.is_open():
            return abs(position.quantity * position.entry_price)
        return 0.0

    def total_notional(self) -> float:
        return sum(abs(p.quantity * p.entry_price) for p in self.get_open_positions())

    def update_on_fill(self, fill: OrderFill, side: Side, symbol: str, leverage: int, stop_loss: Optional[float], take_profit: Optional[float]):
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_open():
            pos = Position(
                symbol=symbol,
                side=side,
                entry_price=fill.avg_price,
                quantity=fill.filled_qty,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                opened_at=fill.timestamp,
            )
            self.positions[symbol] = pos
            logger.info("Opened position", extra={"position": pos.model_dump()})
            log_event(
                "POSITION_OPENED",
                symbol=symbol,
                side=side.value,
                entry_price=fill.avg_price,
                qty=fill.filled_qty,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                run_mode=self.run_mode,
                position=pos.model_dump(),
            )
            self._persist_state()
        else:
            if pos.has_added_once:
                logger.warning("Add-on blocked: already added once", extra={"symbol": symbol})
                return
            projected_qty = pos.quantity + fill.filled_qty
            notional = abs(projected_qty * fill.avg_price)
            max_notional = self.equity * MAX_POSITION_PCT
            if max_notional > 0 and notional > max_notional:
                logger.warning(
                    "Add-on blocked: position cap",
                    extra={"symbol": symbol, "notional": notional, "max_notional": max_notional},
                )
                return
            total_qty = pos.quantity + fill.filled_qty
            pos.entry_price = (pos.entry_price * pos.quantity + fill.avg_price * fill.filled_qty) / total_qty
            pos.quantity = total_qty
            pos.has_added_once = True
            logger.info("Added to position", extra={"position": pos.model_dump()})
            self._persist_state()

    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_open():
            return None
        pnl = (exit_price - pos.entry_price) * pos.quantity
        if pos.side == Side.SHORT:
            pnl = -pnl
        pos.realized_pnl += pnl
        pos.closed_at = datetime.utcnow()
        self.equity += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        logger.info("Closed position", extra={"pnl": pnl, "equity": self.equity})
        notional = abs(pos.entry_price * pos.quantity) or 1.0
        pnl_pct = pnl / notional
        log_event(
            "POSITION_CLOSED",
            symbol=symbol,
            side=pos.side.value,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            equity=self.equity,
            run_mode=self.run_mode,
            position=pos.model_dump(),
        )
        self._persist_state()
        return pnl

    def apply_fee(self, amount: float):
        self.equity -= amount
        self.fees_paid += amount
        self._persist_state()

    def _persist_state(self) -> None:
        if not self.state_store:
            return
        try:
            positions_payload = {symbol: pos.model_dump() for symbol, pos in self.positions.items()}
            portfolio_payload = {
                "equity": self.equity,
                "fees_paid": self.fees_paid,
                "consecutive_losses": self.consecutive_losses,
                "run_mode": self.run_mode,
                "updated_at": datetime.utcnow().isoformat(),
                "run_id": self.run_id,
            }
            self.state_store.merge(positions=positions_payload, portfolio=portfolio_payload)
        except Exception as exc:  # noqa: BLE001 - persistence failures must be non-fatal
            logger.warning("Failed to persist position state", extra={"error": str(exc)})


@dataclass
class EquityTracker:
    history: List[float] = field(default_factory=list)

    def record(self, value: float):
        self.history.append(value)

    def max_drawdown(self) -> float:
        peak = 0.0
        max_dd = 0.0
        for v in self.history:
            peak = max(peak, v)
            if peak:
                max_dd = min(max_dd, (v - peak) / peak)
        return max_dd
