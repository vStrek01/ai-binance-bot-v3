from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd

from core.llm_adapter import LLMAdapter
from core.models import Candle, LLMSignal, MarketState, Signal, Side
from infra.logging import logger, log_event

if TYPE_CHECKING:
    from strategies.baseline_rsi_trend import BaselineRSITrend, BaselineDecision
    from strategies.ema_stoch_scalping import EMAStochasticStrategy, ScalpingDecision


@dataclass
class IndicatorConfig:
    fast_ema: int = 13
    slow_ema: int = 34
    rsi_length: int = 14
    rsi_overbought: float = 60.0
    rsi_oversold: float = 40.0
    atr_period: int = 14
    atr_stop: float = 1.6
    atr_target: float = 2.2
    cooldown_bars: int = 2
    hold_bars: int = 90
    pullback_atr_multiplier: float = 1.0
    pullback_rsi_threshold: float = 50.0
    trend_strength_threshold: float = 0.3
    min_confidence: float = 0.55


class Strategy:
    def __init__(
        self,
        indicator_config: IndicatorConfig,
        llm_adapter: Optional[LLMAdapter] = None,
        optimized_params: Optional[Dict[str, Any]] = None,
        strategy_mode: Literal["llm", "baseline", "scalping"] = "llm",
        baseline_strategy: Optional["BaselineRSITrend"] = None,
        scalping_strategy: Optional["EMAStochasticStrategy"] = None,
    ):
        self._base_config = indicator_config
        self.llm_adapter = llm_adapter
        self.optimized_params = optimized_params or {}
        self.config = self._merge_config(self._base_config, self.optimized_params)
        self.strategy_mode = strategy_mode
        self.baseline_strategy = baseline_strategy
        self.scalping_strategy = scalping_strategy

    def _merge_config(self, config: IndicatorConfig, overrides: Dict[str, Any]) -> IndicatorConfig:
        if not overrides:
            return config
        config_dict = asdict(config)
        for key, value in overrides.items():
            if key in config_dict and value is not None:
                config_dict[key] = value
        return IndicatorConfig(**config_dict)

    def _compute_ma_signal(self, candles: List[Candle]) -> Signal:
        closes = [c.close for c in candles]
        if len(closes) < self.config.slow_ema:
            return Signal(action=Side.FLAT, confidence=0.0, reason="insufficient data")

        close_series = pd.Series(closes)
        fast = close_series.rolling(self.config.fast_ema).mean().iloc[-1]
        slow = close_series.rolling(self.config.slow_ema).mean().iloc[-1]

        if np.isnan(fast) or np.isnan(slow):
            return Signal(action=Side.FLAT, confidence=0.0, reason="insufficient data")

        if fast > slow:
            return Signal(action=Side.LONG, confidence=0.6, reason="fast MA above slow")
        if fast < slow:
            return Signal(action=Side.SHORT, confidence=0.6, reason="fast MA below slow")
        return Signal(action=Side.FLAT, confidence=0.0, reason="flat MAs")

    def _fuse_signals(self, indicator_signal: Signal, llm_signal: Optional[LLMSignal]) -> Signal:
        if llm_signal is None:
            return indicator_signal

        # Baseline chooses direction; LLM can only modulate conviction.
        if indicator_signal.action == Side.FLAT:
            return indicator_signal

        if llm_signal.action == Side.FLAT:
            return indicator_signal

        if indicator_signal.action == llm_signal.action:
            boost = min(1.0, indicator_signal.confidence + llm_signal.confidence * 0.25)
            return Signal(action=indicator_signal.action, confidence=boost, reason="agreement")

        # Hard disagreement -> stand down entirely.
        return Signal(action=Side.FLAT, confidence=0.0, reason="llm_conflict")

    def evaluate(self, market_state: MarketState) -> Signal:
        if self.strategy_mode == "baseline" and self.baseline_strategy:
            return self._run_baseline(market_state)
        if self.strategy_mode == "scalping" and self.scalping_strategy:
            return self._run_scalping(market_state)

        indicator_signal = self._compute_ma_signal(market_state.candles)
        llm_signal = None
        if self.llm_adapter:
            context = {
                "symbol": market_state.symbol,
                "last_close": market_state.candles[-1].close,
                "equity": market_state.equity,
            }
            llm_signal = self.llm_adapter.infer(context)

        fused = self._fuse_signals(indicator_signal, llm_signal)
        if fused.confidence < self.config.min_confidence:
            logger.info("Signal confidence too low", extra={"signal": fused.model_dump()})
            return Signal(action=Side.FLAT, confidence=0.0, reason="low confidence")
        return fused

    def _run_baseline(self, market_state: MarketState) -> Signal:
        decision = self.baseline_strategy.evaluate(market_state.candles) if self.baseline_strategy else None
        if decision is None:
            return Signal(action=Side.FLAT, confidence=0.0, reason="baseline_unavailable")
        return self._decision_to_signal(decision, default_reason="baseline_decision")

    def _run_scalping(self, market_state: MarketState) -> Signal:
        position_side = self._resolve_position_state(market_state)
        decision = (
            self.scalping_strategy.evaluate(market_state.candles, position_side=position_side)
            if self.scalping_strategy
            else None
        )
        if decision is None:
            return Signal(action=Side.FLAT, confidence=0.0, reason="scalping_unavailable")
        signal = self._decision_to_signal(decision, default_reason="ema_stoch_decision")
        self._log_scalping_snapshot(market_state, decision)
        return signal

    def _decision_to_signal(self, decision: "BaselineDecision" | "ScalpingDecision", *, default_reason: str) -> Signal:
        reason = getattr(decision, "reason", None) or default_reason
        action_name = getattr(decision, "action", "FLAT") or "FLAT"
        action_name = action_name.upper()
        if action_name in {"CLOSE_LONG", "CLOSE_SHORT"}:
            action = Side.FLAT
        elif action_name == "HOLD":
            action = Side.FLAT
        else:
            try:
                action = Side(action_name)
            except ValueError:
                action = Side.FLAT
        confidence = 1.0 if action != Side.FLAT else 0.0
        size_usd = decision.size_usd if action != Side.FLAT else 0.0
        return Signal(
            action=action,
            confidence=confidence,
            reason=reason,
            stop_loss_pct=decision.sl_pct,
            take_profit_pct=decision.tp_pct,
            size_usd=size_usd,
        )

    def _log_scalping_snapshot(self, market_state: MarketState, decision: "ScalpingDecision") -> None:
        if not self.scalping_strategy or not market_state.candles:
            return
        snapshot = self.scalping_strategy.latest_snapshot or {}
        last_close_time = market_state.candles[-1].close_time if market_state.candles else None
        payload = {
            "symbol": market_state.symbol,
            "close_time": last_close_time.isoformat() if last_close_time else None,
            "last_close": snapshot.get("price"),
            "action": decision.action,
            "reason": decision.reason or "ema_stoch_decision",
            "ema_fast": snapshot.get("ema_fast"),
            "ema_slow": snapshot.get("ema_slow"),
            "stoch_k": snapshot.get("stoch_k"),
            "stoch_k_prev": snapshot.get("stoch_k_prev"),
            "stoch_d": snapshot.get("stoch_d"),
            "stoch_d_prev": snapshot.get("stoch_d_prev"),
            "position_state": snapshot.get("position_state")
            or getattr(decision, "position_state", None),
            "params_preset": snapshot.get("params_preset"),
            "last_trade_bar": snapshot.get("last_trade_bar"),
            "trend_filter_active": snapshot.get("trend_filter_active"),
            "throttle_reason": snapshot.get("throttle_reason"),
            "size_usd": decision.size_usd,
            "sl_pct": decision.sl_pct,
            "tp_pct": decision.tp_pct,
        }
        log_event("SCALPING_SIGNAL", **payload)

    def _resolve_position_state(self, market_state: MarketState) -> Side:
        candidate = market_state.position
        if (
            candidate
            and getattr(candidate, "symbol", None) == market_state.symbol
            and getattr(candidate, "is_open", lambda: False)()
        ):
            return getattr(candidate, "side", Side.FLAT)
        for pos in market_state.open_positions or []:
            if getattr(pos, "symbol", None) != market_state.symbol:
                continue
            if hasattr(pos, "is_open") and not pos.is_open():
                continue
            side = getattr(pos, "side", None)
            if isinstance(side, Side):
                return side
            if isinstance(side, str):
                try:
                    return Side(side)
                except ValueError:
                    continue
        return Side.FLAT
