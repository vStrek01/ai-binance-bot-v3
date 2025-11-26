from exchange.binance_stream import BinanceStream


def test_binance_stream_uses_configured_base_url() -> None:
    base = "wss://demo-stream.binancefuture.com"
    stream = BinanceStream("BTCUSDT", interval="5m", testnet=True, ws_market_url=base)
    assert stream.ws_url.startswith(base.rstrip("/") + "/stream")
    assert "btcusdt@kline_5m" in stream.ws_url
