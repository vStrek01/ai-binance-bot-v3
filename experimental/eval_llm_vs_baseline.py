"""EXPERIMENTAL: Offline evaluator comparing baseline vs LLM strategy modes.

Example:
    # PowerShell
    .\.venv\Scripts\Activate.ps1
    python -m experimental.eval_llm_vs_baseline --data-path data/BTCUSDT_1m.csv --symbol BTCUSDT --max-rows 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from pandas.api.types import is_numeric_dtype

from core.engine import TradingEngine
from core.llm_adapter import LLMAdapter
from core.models import Candle, RiskConfig as EngineRiskConfig
from core.risk import RiskManager
from core.safety import SafetyLimits
from core.state import PositionManager
from core.strategy import IndicatorConfig, Strategy
from infra.config_loader import load_config
from infra.config_schema import AppConfig
from infra.logging import bind_log_context, setup_logging
from strategies.baseline_rsi_trend import BaselineConfig, BaselineRSITrend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs LLM strategies on historical data")
    parser.add_argument("--data-path", required=True, help="CSV file with candles (open_time, open, high, low, close, volume)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--max-rows", type=int, default=5000, help="Maximum candles to load (from the end)")
    parser.add_argument("--start", type=str, help="ISO start time filter", default=None)
    parser.add_argument("--end", type=str, help="ISO end time filter", default=None)
    return parser.parse_args()


def _load_candles(path: Path, symbol: str, max_rows: int, start: str | None, end: str | None) -> List[Candle]:
    df = pd.read_csv(path)
    if "open_time" not in df.columns:
        raise ValueError("CSV must include open_time column")
    if "close_time" not in df.columns:
        df["close_time"] = df["open_time"] + 60_000

    open_time_series = df["open_time"]
    close_time_series = df["close_time"]
    if is_numeric_dtype(open_time_series):
        df["_open_dt"] = pd.to_datetime(open_time_series, unit="ms", utc=True)
    else:
        df["_open_dt"] = pd.to_datetime(open_time_series, utc=True)

    if is_numeric_dtype(close_time_series):
        df["_close_dt"] = pd.to_datetime(close_time_series, unit="ms", utc=True)
    else:
        df["_close_dt"] = pd.to_datetime(close_time_series, utc=True)

    if start:
        start_ts = pd.to_datetime(start, utc=True)
        df = df[df["_open_dt"] >= start_ts]
    if end:
        end_ts = pd.to_datetime(end, utc=True)
        df = df[df["_open_dt"] <= end_ts]

    if max_rows:
        df = df.tail(max_rows)

    candles: List[Candle] = []
    for row in df.itertuples(index=False):
        open_time = getattr(row, "_open_dt").to_pydatetime()
        close_time = getattr(row, "_close_dt").to_pydatetime()
        candles.append(
            Candle(
                symbol=symbol,
                open_time=open_time,
                close_time=close_time,
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=float(getattr(row, "volume", 0.0)),
            )
        )
    if not candles:
        raise ValueError("No candles loaded from dataset")
    return candles


def _build_safety_limits(config: AppConfig) -> SafetyLimits:
    risk = config.risk
    return SafetyLimits(
        max_daily_drawdown_pct=risk.max_daily_drawdown_pct,
        max_total_notional_usd=risk.max_notional_global,
        max_consecutive_losses=risk.max_consecutive_losses,
    )


def _build_risk_manager(config: AppConfig, safety_limits: SafetyLimits) -> RiskManager:
    base_risk = EngineRiskConfig()
    overrides = {
        "max_risk_per_trade_pct": config.risk.risk_per_trade_pct / 100.0,
        "max_daily_drawdown_pct": config.risk.max_daily_drawdown_pct / 100.0,
        "max_symbol_notional_usd": config.risk.max_notional_per_symbol,
    }
    risk_cfg = base_risk.model_copy(update=overrides)
    return RiskManager(risk_cfg, safety_limits=safety_limits)


def _build_strategy(config: AppConfig, mode: str) -> Strategy:
    strategy_cfg = config.strategy
    indicator_cfg = IndicatorConfig(
        fast_ema=strategy_cfg.ema_fast,
        slow_ema=strategy_cfg.ema_slow,
        rsi_length=strategy_cfg.rsi_length,
        rsi_overbought=float(strategy_cfg.rsi_overbought),
        rsi_oversold=float(strategy_cfg.rsi_oversold),
        atr_period=strategy_cfg.atr_length,
        atr_stop=strategy_cfg.atr_multiplier,
        atr_target=strategy_cfg.atr_multiplier,
    )

    baseline_strategy = None
    llm_adapter = None
    if mode == "baseline":
        baseline_strategy = BaselineRSITrend(
            BaselineConfig(
                ma_length=strategy_cfg.ema_slow,
                rsi_length=strategy_cfg.rsi_length,
                rsi_oversold=float(strategy_cfg.rsi_oversold),
                rsi_overbought=float(strategy_cfg.rsi_overbought),
            )
        )
    elif config.llm.enabled:
        llm_adapter = LLMAdapter()

    return Strategy(
        indicator_cfg,
        llm_adapter=llm_adapter,
        strategy_mode=mode,
        baseline_strategy=baseline_strategy,
    )


def _run_backtest(candles: List[Candle], config: AppConfig, mode: str) -> dict:
    safety_limits = _build_safety_limits(config)
    risk_manager = _build_risk_manager(config, safety_limits)
    position_manager = PositionManager(run_mode="backtest")
    strategy = _build_strategy(config, mode)
    engine = TradingEngine(
        strategy,
        risk_manager,
        position_manager,
        safety_limits=safety_limits,
        run_mode="backtest",
    )
    return engine.run_backtest(candles)


def main() -> None:
    args = _parse_args()
    setup_logging(state_file="logs/dashboard_state.json")
    config = load_config(mode_override="backtest")
    bind_log_context(run_mode=config.run_mode, run_id="eval_llm_vs_baseline")
    candles = _load_candles(Path(args.data_path), args.symbol, args.max_rows, args.start, args.end)

    baseline_result = _run_backtest(candles, config, "baseline")
    llm_result = _run_backtest(candles, config, "llm")

    comparison = {
        "baseline": baseline_result.get("summary", {}),
        "llm": llm_result.get("summary", {}),
    }
    print(json.dumps(comparison, indent=2, default=str))


if __name__ == "__main__":
    main()
