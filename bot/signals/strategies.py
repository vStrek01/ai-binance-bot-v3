"""Strategy definitions shared across backtest, execution, and RL."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from bot.core.config import BotConfig
from bot.signals import indicators
from infra.logging import log_event


@dataclass(slots=True)
class StrategyParameters:
    fast_ema: int
    slow_ema: int
    rsi_length: int
    rsi_overbought: float
    rsi_oversold: float
    atr_period: int
    atr_stop: float
    atr_target: float
    cooldown_bars: int
    hold_bars: int
    min_volatility_pct: float = 0.05
    max_spread_pct: float = 0.25
    trend_confirm_bars: int = 0


@dataclass(slots=True)
class StrategySignal:
    index: int
    direction: int
    entry_price: float
    stop_loss: float
    take_profit: float
    indicators: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 1.0

    @property
    def action(self) -> str:
        return "LONG" if self.direction == 1 else "SHORT"

    def to_decision(self, *, symbol: Optional[str] = None, interval: Optional[str] = None, confidence: Optional[float] = None) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "interval": interval,
            "action": self.action,
            "confidence": float(confidence if confidence is not None else self.confidence),
            "reason": self.reason or "strategy_signal",
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "indicators": dict(self.indicators),
        }


class EmaRsiAtrStrategy:
    def __init__(
        self,
        params: StrategyParameters,
        *,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        run_mode: str = "backtest",
    ) -> None:
        self.params = params
        self.symbol = symbol
        self.interval = interval
        self.run_mode = run_mode
        self._telemetry_enabled = run_mode in {"demo-live", "live"}
        self._last_snapshot: Optional[Dict[str, float]] = None

    def _annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy().reset_index(drop=True)
        enriched["ema_fast"] = indicators.ema(enriched["close"], self.params.fast_ema)
        enriched["ema_slow"] = indicators.ema(enriched["close"], self.params.slow_ema)
        enriched["rsi"] = indicators.rsi(enriched["close"], self.params.rsi_length)
        enriched["atr"] = indicators.atr(enriched, self.params.atr_period)
        enriched["atr_pct"] = (enriched["atr"] / enriched["close"].replace(0, pd.NA) * 100).fillna(0)
        enriched["spread_pct"] = ((enriched["high"] - enriched["low"]).abs() / enriched["close"].replace(0, pd.NA) * 100).fillna(0)
        trend_window = max(self.params.trend_confirm_bars or 0, self.params.slow_ema)
        if trend_window > 0:
            enriched["trend_ema"] = indicators.ema(enriched["close"], trend_window)
        else:
            enriched["trend_ema"] = pd.NA
        return enriched

    def generate_signals(self, frame: pd.DataFrame) -> List[StrategySignal]:
        annotated = self._annotate(frame)
        signals: List[StrategySignal] = []
        cooldown = 0
        total_rows = len(annotated)
        for idx, row in enumerate(annotated.itertuples(), start=0):
            if cooldown > 0:
                remaining = cooldown
                cooldown -= 1
                if idx == total_rows - 1:
                    self._log_veto("cooldown", "cooldown_active", bars_remaining=remaining)
                    self._record_snapshot(row, remaining)
                continue
            if pd.isna(row.atr):
                continue
            price = float(row.close)
            ema_fast = float(row.ema_fast)
            ema_slow = float(row.ema_slow)
            rsi_value = float(row.rsi)
            atr_value = float(row.atr)
            atr_pct = float(getattr(row, "atr_pct", 0.0) or 0.0)
            spread_pct = float(getattr(row, "spread_pct", 0.0) or 0.0)
            if atr_pct < self.params.min_volatility_pct:
                if idx == total_rows - 1:
                    self._record_snapshot(row, cooldown, atr_pct=atr_pct, spread_pct=spread_pct)
                self._log_veto(
                    "volatility",
                    "below_threshold",
                    atr_pct=atr_pct,
                    threshold=self.params.min_volatility_pct,
                )
                continue
            if self.params.max_spread_pct > 0 and spread_pct > self.params.max_spread_pct:
                if idx == total_rows - 1:
                    self._record_snapshot(row, cooldown, atr_pct=atr_pct, spread_pct=spread_pct)
                self._log_veto(
                    "spread",
                    "exceeds_threshold",
                    spread_pct=spread_pct,
                    threshold=self.params.max_spread_pct,
                )
                continue
            go_long = ema_fast > ema_slow and rsi_value <= self.params.rsi_oversold
            go_short = ema_fast < ema_slow and rsi_value >= self.params.rsi_overbought
            direction = 0
            stop = take_profit = 0.0
            if go_long:
                direction = 1
                stop = price - atr_value * self.params.atr_stop
                target = price + atr_value * self.params.atr_target
            elif go_short:
                direction = -1
                stop = price + atr_value * self.params.atr_stop
                target = price - atr_value * self.params.atr_target
            if direction == 0:
                if idx == total_rows - 1:
                    self._record_snapshot(row, cooldown, atr_pct=atr_pct, spread_pct=spread_pct)
                continue

            trend_value = getattr(row, "trend_ema", None)
            if not self._trend_confirmed(direction, price, trend_value):
                self._log_veto(
                    "trend",
                    "misaligned",
                    trend=float(trend_value) if trend_value is not None else None,
                    price=price,
                    direction="LONG" if direction == 1 else "SHORT",
                )
                if idx == total_rows - 1:
                    self._record_snapshot(row, cooldown, atr_pct=atr_pct, spread_pct=spread_pct, trend=trend_value)
                continue

            signals.append(
                StrategySignal(
                    idx,
                    direction,
                    price,
                    stop,
                    target,
                    indicators={
                        "ema_fast": ema_fast,
                        "ema_slow": ema_slow,
                        "rsi": rsi_value,
                        "atr": atr_value,
                        "atr_pct": atr_pct,
                        "spread_pct": spread_pct,
                    },
                    reason="ema_trend_rsi_pullback",
                )
            )
            cooldown = self.params.cooldown_bars
            if idx == total_rows - 1:
                self._record_snapshot(row, cooldown, atr_pct=atr_pct, spread_pct=spread_pct, trend=trend_value)
        return signals

    def latest_snapshot(self) -> Optional[Dict[str, float]]:
        return self._last_snapshot

    def _record_snapshot(self, row: pd.Series | pd.Index, cooldown: int, **extras: Any) -> None:
        try:
            price = float(row.close)
        except AttributeError:  # pragma: no cover - defensive
            price = None
        snapshot = {
            "last_close": price,
            "ema_fast": float(row.ema_fast) if hasattr(row, "ema_fast") else None,
            "ema_slow": float(row.ema_slow) if hasattr(row, "ema_slow") else None,
            "rsi": float(row.rsi) if hasattr(row, "rsi") else None,
            "atr": float(row.atr) if hasattr(row, "atr") else None,
            "cooldown_bars": int(max(cooldown, 0)),
            "atr_pct": float(getattr(row, "atr_pct", 0.0) or 0.0),
            "spread_pct": float(getattr(row, "spread_pct", 0.0) or 0.0),
            "trend": float(getattr(row, "trend_ema", 0.0) or 0.0) if hasattr(row, "trend_ema") else None,
        }
        for key, value in extras.items():
            snapshot[key] = value
        self._last_snapshot = snapshot

    def _log_veto(self, stage: str, reason: str, **details: Any) -> None:
        if not self._telemetry_enabled:
            return
        log_event(
            "STRATEGY_VETO",
            run_mode=self.run_mode,
            symbol=self.symbol,
            interval=self.interval,
            reason=reason,
            stage=stage,
            **{k: v for k, v in details.items() if v is not None},
        )

    def _trend_confirmed(self, direction: int, price: float, trend_value: Any) -> bool:
        if not self.params.trend_confirm_bars:
            return True
        try:
            trend = float(trend_value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return True
        if pd.isna(trend):
            return True
        if direction == 1:
            return price >= trend
        return price <= trend


def build_parameters(cfg: BotConfig, overrides: Optional[Dict[str, float | int]] = None) -> StrategyParameters:
    settings = cfg.strategy.default_parameters.copy()
    if cfg.run_mode == "demo-live":
        settings.update(cfg.strategy.demo_live_overrides)
    if overrides:
        settings.update(overrides)
    baseline = cfg.baseline_strategy
    return StrategyParameters(
        fast_ema=int(settings["fast_ema"]),
        slow_ema=int(settings["slow_ema"]),
        rsi_length=int(settings.get("rsi_length", 14)),
        rsi_overbought=float(settings["rsi_overbought"]),
        rsi_oversold=float(settings["rsi_oversold"]),
        atr_period=int(settings.get("atr_period", 14)),
        atr_stop=float(settings["atr_stop"]),
        atr_target=float(settings["atr_target"]),
        cooldown_bars=int(settings.get("cooldown_bars", 3)),
        hold_bars=int(settings.get("hold_bars", 120)),
        min_volatility_pct=float(settings.get("min_volatility_pct", baseline.min_atr_pct)),
        max_spread_pct=float(settings.get("max_spread_pct", baseline.max_atr_pct)),
        trend_confirm_bars=int(settings.get("trend_confirm_bars", baseline.slow_ema * 2)),
    )


__all__ = ["StrategyParameters", "StrategySignal", "EmaRsiAtrStrategy", "build_parameters"]
