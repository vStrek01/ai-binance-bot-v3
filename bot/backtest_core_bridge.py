"""Bridge helpers that allow the bot CLI to run the core EMA backtester."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from bot.core.config import BotConfig
from bot.signals.strategies import StrategyParameters
from bot.strategy_mode import LLM_MODE, SCALPING_MODE, build_scalping_config, resolve_strategy_mode
from core.engine import TradingEngine
from core.models import Candle, RiskConfig
from core.risk import RiskManager
from core.state import PositionManager
from core.strategy import IndicatorConfig, Strategy
from core.safety import SafetyLimits
from infra.logging import logger
from infra.persistence import save_backtest_results
from strategies.baseline_rsi_trend import BaselineConfig, BaselineRSITrend
from strategies.ema_stoch_scalping import EMAStochasticStrategy


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()  # type: ignore[no-any-return]
    return pd.Timestamp(value).to_pydatetime()  # type: ignore[arg-type]


def _frame_to_candles(frame: pd.DataFrame, symbol: str) -> List[Candle]:
    candles: List[Candle] = []
    for row in frame.itertuples(index=False):
        candles.append(
            Candle(
                symbol=symbol,
                open_time=_timestamp(row.open_time),
                close_time=_timestamp(row.close_time),
                open=float(row.open),  # type: ignore[arg-type]
                high=float(row.high),  # type: ignore[arg-type]
                low=float(row.low),  # type: ignore[arg-type]
                close=float(row.close),  # type: ignore[arg-type]
                volume=float(row.volume),  # type: ignore[arg-type]
            )
        )
    return candles


def _build_risk_config(cfg: BotConfig) -> RiskConfig:
    slippage = cfg.backtest.slippage_bps / 10_000
    return RiskConfig(
        max_risk_per_trade_pct=cfg.risk.per_trade_risk,
        max_daily_drawdown_pct=cfg.risk.max_daily_loss_pct or 0.04,
        max_open_positions=cfg.risk.max_concurrent_symbols or 1,
        max_leverage=int(cfg.risk.leverage),
        taker_fee_rate=cfg.risk.taker_fee,
        maker_fee_rate=cfg.risk.maker_fee,
        slippage=slippage,
    )


def _build_strategy(
    cfg: BotConfig,
    params: StrategyParameters,
    params_dict: Optional[Dict[str, Any]] = None,
    strategy_mode: Optional[str] = None,
) -> Strategy:
    payload = params_dict or asdict(params)
    defaults = IndicatorConfig()
    indicator_cfg = IndicatorConfig(
        fast_ema=int(payload.get("fast_ema", params.fast_ema)),
        slow_ema=int(payload.get("slow_ema", params.slow_ema)),
        rsi_length=int(payload.get("rsi_length", params.rsi_length)),
        rsi_overbought=float(payload.get("rsi_overbought", params.rsi_overbought)),
        rsi_oversold=float(payload.get("rsi_oversold", params.rsi_oversold)),
        atr_period=int(payload.get("atr_period", params.atr_period)),
        atr_stop=float(payload.get("atr_stop", params.atr_stop)),
        atr_target=float(payload.get("atr_target", params.atr_target)),
        cooldown_bars=int(payload.get("cooldown_bars", params.cooldown_bars)),
        hold_bars=int(payload.get("hold_bars", params.hold_bars)),
        min_confidence=float(payload.get("min_confidence", defaults.min_confidence)),
        pullback_atr_multiplier=float(payload.get("pullback_atr_multiplier", defaults.pullback_atr_multiplier)),
        pullback_rsi_threshold=float(payload.get("pullback_rsi_threshold", defaults.pullback_rsi_threshold)),
        trend_strength_threshold=float(payload.get("trend_strength_threshold", defaults.trend_strength_threshold)),
    )
    mode = resolve_strategy_mode(cfg, override=strategy_mode)
    allowed_modes = {"llm", "baseline", SCALPING_MODE}
    if mode not in allowed_modes:
        mode = LLM_MODE

    baseline_strategy = None
    scalping_strategy = None
    if mode == "baseline":
        baseline_strategy = BaselineRSITrend()
    elif mode == SCALPING_MODE:
        scalping_strategy = EMAStochasticStrategy(build_scalping_config(cfg))

    return Strategy(
        indicator_cfg,
        llm_adapter=None,
        optimized_params=payload,
        strategy_mode=mode,
        baseline_strategy=baseline_strategy,
        scalping_strategy=scalping_strategy,
    )


def _build_safety_limits(cfg: BotConfig) -> SafetyLimits:
    max_daily_pct = float((cfg.risk.max_daily_loss_pct or 0.04) * 100)
    leverage = max(cfg.risk.leverage, 1.0)
    exposure_cap = max(cfg.risk.max_account_exposure, 0.01)
    total_notional = cfg.backtest.initial_balance * leverage * exposure_cap
    configured_global_cap = cfg.risk.max_notional_global or 0.0
    max_total_notional = max(configured_global_cap, total_notional, cfg.backtest.initial_balance)
    loss_cap = cfg.risk.max_consecutive_losses or 0
    if loss_cap <= 0:
        loss_cap = max(cfg.risk.max_concurrent_symbols, 1)
    return SafetyLimits(
        max_daily_drawdown_pct=max(max_daily_pct, 1.0),
        max_total_notional_usd=max_total_notional,
        max_consecutive_losses=loss_cap,
    )


def run_core_backtest(
    cfg: BotConfig,
    symbol: str,
    interval: str,
    frame: pd.DataFrame,
    params: StrategyParameters,
    params_dict: Optional[Dict[str, Any]] = None,
    params_source: str = "default",
    strategy_mode: Optional[str] = None,
) -> dict:
    dataset = frame.tail(cfg.backtest.max_bars).reset_index(drop=True)
    if dataset.empty:
        raise ValueError("Backtest dataset is empty after applying max_bars")

    candles = _frame_to_candles(dataset, symbol)
    strategy = _build_strategy(cfg, params, params_dict, strategy_mode=strategy_mode)
    safety_limits = _build_safety_limits(cfg)
    risk_manager = RiskManager(_build_risk_config(cfg), safety_limits=safety_limits)
    position_manager = PositionManager(equity=cfg.backtest.initial_balance, run_mode="backtest")
    engine = TradingEngine(
        strategy,
        risk_manager,
        position_manager,
        safety_limits=safety_limits,
        run_mode="backtest",
    )

    params_payload = params_dict or asdict(params)
    logger.info(
        "Running core backtest",
        extra={"symbol": symbol, "params": params_payload, "source": params_source},
    )
    result = engine.run_backtest(candles)
    summary = result.get("summary", {})
    result["strategy_params"] = params_payload
    result["strategy_params_source"] = params_source
    payload = {
        "symbol": symbol,
        "interval": interval,
        "generated_at": datetime.utcnow().isoformat(),
        "strategy_params": params_payload,
        "strategy_params_source": params_source,
        "summary": summary,
        "metrics": result.get("metrics", {}),
        "equity_curve": result.get("equity_curve", []),
        "trades": result.get("trades", []),
    }
    output_path = save_backtest_results(payload, symbol=symbol, interval=interval)
    logger.info("Saved backtest results", extra={"path": str(output_path)})
    return result


__all__ = ["run_core_backtest"]
