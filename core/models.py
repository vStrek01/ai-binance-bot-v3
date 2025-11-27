from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from infra.pydantic_guard import ensure_pydantic_v2
from pydantic import BaseModel, Field, field_validator

ensure_pydantic_v2()


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Candle(BaseModel):
    symbol: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class Position(BaseModel):
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    leverage: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    has_added_once: bool = False

    def is_open(self) -> bool:
        return self.closed_at is None


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderRequest(BaseModel):
    symbol: str
    side: Side
    order_type: OrderType = OrderType.MARKET
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reduce_only: bool = False
    leverage: int = 1


class OrderFill(BaseModel):
    order_id: str
    status: str
    filled_qty: float
    avg_price: float
    timestamp: datetime
    client_order_id: Optional[str] = None


class Signal(BaseModel):
    action: Side
    confidence: float = 0.0
    reason: str = ""
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    size_usd: Optional[float] = None

    @field_validator("confidence")
    @classmethod
    def _bound_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class LLMSignal(Signal):
    pass


class MarketState(BaseModel):
    symbol: str
    candles: List[Candle]
    position: Optional[Position] = None
    equity: float = 0.0
    open_positions: List[Position] = Field(default_factory=list)


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = 0.01
    max_daily_drawdown_pct: float = 0.04
    max_open_positions: int = 3
    max_leverage: int = 5
    taker_fee_rate: float = 0.0006
    maker_fee_rate: float = 0.0002
    slippage: float = 0.0005
    max_symbol_notional_usd: float = 5_000.0
    max_total_notional_usd: float = 10_000.0
    min_order_notional_usd: float = 10.0
    max_trades_per_day: int = 500
    max_commission_pct_per_day: float = 1.0
    max_consecutive_losses: int = 5

    @field_validator(
        "max_risk_per_trade_pct",
        "max_daily_drawdown_pct",
        "taker_fee_rate",
        "maker_fee_rate",
        "slippage",
        "max_symbol_notional_usd",
        "max_total_notional_usd",
        "min_order_notional_usd",
        "max_commission_pct_per_day",
    )
    @classmethod
    def _positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Risk parameters must be non-negative")
        return v

    @field_validator("max_trades_per_day", "max_consecutive_losses", "max_open_positions")
    @classmethod
    def _positive_ints(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Risk limits must be non-negative")
        return value
