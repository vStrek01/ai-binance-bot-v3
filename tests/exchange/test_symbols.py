import pytest

from exchange.symbols import SymbolResolver


class _DummyClient:
    def __init__(self, payload):
        self._payload = payload

    def get_exchange_info(self):  # pragma: no cover - simple stub
        return self._payload


def _sample_payload():
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "contractType": "PERPETUAL",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
        ]
    }


def test_symbol_resolver_rounds_and_validates():
    resolver = SymbolResolver(_DummyClient(_sample_payload()), symbols=["BTCUSDT"])
    resolver.refresh()
    info = resolver.get("btcusdt")

    assert info.round_price(27234.189) == pytest.approx(27234.1)
    assert info.round_qty(0.12356) == pytest.approx(0.123)
    assert info.validate_notional(20_000, 0.001)
    assert not info.validate_notional(1000, 0.0001)


def test_symbol_resolver_missing_symbol_raises():
    resolver = SymbolResolver(_DummyClient(_sample_payload()), symbols=["ETHUSDT"])
    with pytest.raises(ValueError):
        resolver.refresh()
