import pandas as pd
import pytest

from bot.backtester import PortfolioBacktester, PortfolioSlice, TradeResult
from bot.core.config import load_config
from bot.exchange_info import ExchangeInfoManager, SymbolFilters
from bot.signals.strategies import StrategyParameters


@pytest.fixture
def sample_config(tmp_path):
    cfg = load_config(base_dir=tmp_path)
    return cfg


def _filters():
    return SymbolFilters(min_qty=0.001, min_notional=5.0, step_size=0.001, tick_size=0.01, max_leverage=50)


def _strategy_params():
    return StrategyParameters(
        fast_ema=8,
        slow_ema=21,
        rsi_length=14,
        rsi_overbought=60,
        rsi_oversold=40,
        atr_period=14,
        atr_stop=1.6,
        atr_target=2.0,
        cooldown_bars=2,
        hold_bars=50,
    )


def _candles():
    data = {
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100.5, 101.5, 102.5],
        "volume": [1_000, 1_200, 1_400],
    }
    return pd.DataFrame(data)


def test_portfolio_backtester_combines_results(sample_config, monkeypatch):
    exchange = ExchangeInfoManager(sample_config, client=None, prefetched={"BTCUSDT": _filters(), "ETHUSDT": _filters()})
    portfolio = PortfolioBacktester(sample_config, exchange)

    calls = []

    def fake_run(self, symbol, timeframe, candles, params, *, initial_balance=None):  # noqa: ARG001
        calls.append((symbol, initial_balance))
        pnl = 10.0 if symbol == "BTCUSDT" else -5.0
        trade = TradeResult(
            symbol=symbol,
            timeframe=timeframe,
            direction=1,
            quantity=1.0,
            entry_price=100.0,
            exit_price=100.0 + pnl,
            pnl=pnl,
            reason="target",
            opened_at="2023-01-01T00:00:00Z",
            closed_at="2023-01-01T00:00:0{}Z".format(2 if symbol == "BTCUSDT" else 1),
            fees=0.0,
        )
        ending = float(initial_balance or sample_config.backtest.initial_balance) + pnl
        metrics = {
            "total_pnl": pnl,
            "trades": 1,
            "win_rate": 1.0 if pnl > 0 else 0.0,
            "profit_factor": float("inf") if pnl > 0 else 0.0,
            "expectancy": pnl,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "ending_balance": ending,
        }
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "trades": [trade],
            "metrics": metrics,
            "equity_curve": [initial_balance or 0.0, ending],
            "realism": {"mode": "standard"},
        }

    monkeypatch.setattr("bot.backtester.Backtester.run", fake_run)

    slices = [
        PortfolioSlice("BTCUSDT", "1m", _candles(), _strategy_params(), weight=2.0),
        PortfolioSlice("ETHUSDT", "1m", _candles(), _strategy_params(), weight=1.0),
    ]
    result = portfolio.run(slices, initial_balance=1_000.0)

    assert len(result["slices"]) == 2
    assert pytest.approx(result["slices"][0]["allocation"]) == pytest.approx(1000 * (2 / 3))
    assert pytest.approx(result["metrics"]["total_pnl"], rel=1e-6) == 5.0
    assert pytest.approx(result["metrics"]["ending_balance"], rel=1e-6) == 1_005.0
    assert len(result["trades"]) == 2
    assert [trade.symbol for trade in result["trades"]] == ["ETHUSDT", "BTCUSDT"]
    assert calls[0][1] != calls[1][1]