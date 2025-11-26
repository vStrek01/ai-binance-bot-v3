from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from core.models import OrderFill, Position, Side
from infra.logging import logger

MAX_POSITION_PCT = 0.2


@dataclass
class PositionManager:
    positions: Dict[str, Position] = field(default_factory=dict)
    equity: float = 10_000.0
    fees_paid: float = 0.0

    def get_open_positions(self) -> List[Position]:
        return [p for p in self.positions.values() if p.is_open()]

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

    def close_position(self, symbol: str, exit_price: float):
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_open():
            return
        pnl = (exit_price - pos.entry_price) * pos.quantity
        if pos.side == Side.SHORT:
            pnl = -pnl
        pos.realized_pnl += pnl
        pos.closed_at = datetime.utcnow()
        self.equity += pnl
        logger.info("Closed position", extra={"pnl": pnl, "equity": self.equity})

    def apply_fee(self, amount: float):
        self.equity -= amount
        self.fees_paid += amount


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
