from __future__ import annotations

from decimal import Decimal

import pytest

from bot.core.config import load_config
from bot.execution.exchange import ExchangeInfoManager, SymbolFilters
from exchange.symbols import SymbolInfo, SymbolResolver


TEST_SYMBOL = "BTCUSDT"


class DummyExchangeClient:
    def __init__(self, symbol_entry: dict, leverage_payload: list[dict]) -> None:
        self._entry = symbol_entry
        self._leverage = leverage_payload

    def exchange_info(self) -> dict:
        return {"symbols": [self._entry]}

    def leverage_bracket(self) -> list[dict]:
        return self._leverage

    def get_exchange_info(self) -> dict:
        return {"symbols": [self._entry]}


def _sample_symbol_info() -> SymbolInfo:
    return SymbolInfo(
        symbol=TEST_SYMBOL,
        base_asset="BTC",
        quote_asset="USDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("5"),
        min_qty=Decimal("0.001"),
        min_price=Decimal("0"),
        max_price=Decimal("0"),
        multiplier_up=Decimal("1.05"),
        multiplier_down=Decimal("0.95"),
        max_leverage=75,
    )


def _symbol_entry() -> dict:
    return {
        "symbol": TEST_SYMBOL,
        "baseAsset": "BTC",
        "quoteAsset": "USDT",
        "contractType": "PERPETUAL",
        "defaultLeverage": 50,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "0", "maxPrice": "0"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
            {"filterType": "PERCENT_PRICE", "multiplierUp": "1.05", "multiplierDown": "0.95"},
        ],
    }


def _leverage_payload() -> list[dict]:
    return [{"symbol": TEST_SYMBOL, "brackets": [{"initialLeverage": 125}]}]


def test_symbol_filters_share_symbol_info_rounding() -> None:
    info = _sample_symbol_info()
    filters = SymbolFilters.from_symbol_info(info)

    price = 43210.9876
    qty = 0.123456

    assert filters.adjust_price(price) == pytest.approx(info.round_price(price))
    assert filters.adjust_quantity(qty) == pytest.approx(info.round_qty(qty))
    assert filters.min_notional == pytest.approx(float(info.min_notional))


def test_exchange_manager_aligns_with_symbol_resolver(tmp_path) -> None:
    cfg = load_config(base_dir=tmp_path)
    client = DummyExchangeClient(_symbol_entry(), _leverage_payload())

    manager = ExchangeInfoManager(cfg, client=client)
    manager.refresh(force=True)

    resolver = SymbolResolver(client, symbols=[TEST_SYMBOL], ttl_seconds=0)
    resolver.refresh(force=True)

    filters = manager.get_filters(TEST_SYMBOL)
    assert filters is not None

    info = resolver.get(TEST_SYMBOL)

    price = 20123.987
    qty = 0.876543

    assert filters.adjust_price(price) == pytest.approx(info.round_price(price))
    assert filters.adjust_quantity(qty) == pytest.approx(info.round_qty(qty))
    assert filters.max_leverage == pytest.approx(info.max_leverage)

    snapshot = manager.snapshot()[TEST_SYMBOL]
    assert "symbol_info" in snapshot
    serialized = snapshot["symbol_info"]
    restored = SymbolInfo.from_dict(serialized)
    assert restored.symbol == TEST_SYMBOL
    assert restored.tick_size == info.tick_size