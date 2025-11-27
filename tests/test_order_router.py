import asyncio
from decimal import Decimal
from typing import Any, Dict, cast

import pytest

from core.models import OrderRequest, OrderType, Side
from exchange.binance_client import BinanceClient
from exchange.order_router import OrderRouter
from exchange.symbols import SymbolInfo, SymbolResolver


class _DummyClient:
    def __init__(self):
        self.requests: list[Dict[str, Any]] = []

    def check_time_drift(self) -> None:  # pragma: no cover - trivial stub
        return None

    def place_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.requests.append(params)
        return {
            "orderId": 1,
            "status": "FILLED",
            "executedQty": params["quantity"],
            "avgPrice": params.get("price", 100.0),
            "transactTime": 0,
        }


class _ResolverClient:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def get_exchange_info(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return self._payload


def _resolver(symbol_info: SymbolInfo) -> SymbolResolver:
    payload = {
        "symbols": [
            {
                "symbol": symbol_info.symbol,
                "contractType": "PERPETUAL",
                "baseAsset": symbol_info.base_asset,
                "quoteAsset": symbol_info.quote_asset,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": str(symbol_info.tick_size)},
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": str(symbol_info.min_qty),
                        "stepSize": str(symbol_info.step_size),
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": str(symbol_info.min_notional)},
                ],
            }
        ]
    }
    resolver = SymbolResolver(_ResolverClient(payload), symbols=[symbol_info.symbol])
    resolver.refresh()
    return resolver


def test_order_router_normalizes_with_symbol_filters():
    info = SymbolInfo(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("5"),
        min_qty=Decimal("0.001"),
    )
    client = cast(BinanceClient, _DummyClient())
    router = OrderRouter(client, symbol_resolver=_resolver(info))
    order = OrderRequest(
        symbol="BTCUSDT",
        side=Side.LONG,
        order_type=OrderType.LIMIT,
        quantity=0.123456,
        price=20500.187,
        stop_loss=20400.111,
        take_profit=21000.999,
    )

    fill = asyncio.run(router.execute(order))
    assert fill is not None
    assert client.requests[0]["price"] == pytest.approx(20500.1)
    assert client.requests[0]["quantity"] == pytest.approx(0.123)


def test_order_router_rejects_below_notional():
    info = SymbolInfo(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("50"),
        min_qty=Decimal("0.001"),
    )
    client = cast(BinanceClient, _DummyClient())
    router = OrderRouter(client, symbol_resolver=_resolver(info))
    tiny_order = OrderRequest(
        symbol="BTCUSDT",
        side=Side.LONG,
        order_type=OrderType.LIMIT,
        quantity=0.0001,
        price=1.0,
    )

    fill = asyncio.run(router.execute(tiny_order))
    assert fill is None
    assert client.requests == []
