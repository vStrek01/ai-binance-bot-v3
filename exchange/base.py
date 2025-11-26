from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import OrderFill, OrderRequest, Position


class Exchange(ABC):
    """Common interface for live and simulated exchanges."""

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> Optional[OrderFill]:  # pragma: no cover - interface definition
        raise NotImplementedError

    @abstractmethod
    async def cancel_order(self, symbol: str, client_order_id: str) -> bool:  # pragma: no cover - interface definition
        raise NotImplementedError

    @abstractmethod
    def get_balance(self) -> float:  # pragma: no cover - interface definition
        raise NotImplementedError

    @abstractmethod
    def get_open_positions(self) -> List[Position]:  # pragma: no cover - interface definition
        raise NotImplementedError

    @abstractmethod
    def get_open_orders(self) -> List[dict]:  # pragma: no cover - interface definition
        raise NotImplementedError
