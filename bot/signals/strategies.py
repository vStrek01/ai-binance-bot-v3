"""Strategy definitions shared across backtest, execution, and RL."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


@dataclass(slots=True)
class StrategySignal:
    index: int
    direction: int
    entry_price: float
    stop_loss: float
    take_profit: float
    indicators: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


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
                    self._log_veto("cooldown_active", {"bars_remaining": remaining})
                    self._record_snapshot(row, remaining)
                continue
            if pd.isna(row.atr):
                continue
            price = float(row.close)
            ema_fast = float(row.ema_fast)
            ema_slow = float(row.ema_slow)
            rsi_value = float(row.rsi)
            atr_value = float(row.atr)
            go_long = ema_fast > ema_slow and rsi_value <= self.params.rsi_oversold
            go_short = ema_fast < ema_slow and rsi_value >= self.params.rsi_overbought
            if go_long:
                stop = price - atr_value * self.params.atr_stop
                target = price + atr_value * self.params.atr_target
                signals.append(
                    StrategySignal(
                        idx,
                        1,
                        price,
                        stop,
                        target,
                        indicators={
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "rsi": rsi_value,
                            "atr": atr_value,
                        },
                        reason="ema_trend_rsi_pullback",
                    )
                )
                cooldown = self.params.cooldown_bars
            elif go_short:
                stop = price + atr_value * self.params.atr_stop
                target = price - atr_value * self.params.atr_target
                signals.append(
                    StrategySignal(
                        idx,
                        -1,
                        price,
                        stop,
                        target,
                        indicators={
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "rsi": rsi_value,
                            "atr": atr_value,
                        },
                        reason="ema_trend_rsi_pullback",
                    )
                )
                cooldown = self.params.cooldown_bars
            if idx == total_rows - 1:
                self._record_snapshot(row, cooldown)
        return signals

    def latest_snapshot(self) -> Optional[Dict[str, float]]:
        return self._last_snapshot

    def _record_snapshot(self, row: pd.Series | pd.Index, cooldown: int) -> None:
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
        }
        self._last_snapshot = snapshot

    def _log_veto(self, reason: str, details: Dict[str, float]) -> None:
        if not self._telemetry_enabled:
            return
        log_event(
            "STRATEGY_VETO",
            run_mode=self.run_mode,
            symbol=self.symbol,
            interval=self.interval,
            reason=reason,
            stage="cooldown",
            **details,
        )


def build_parameters(cfg: BotConfig, overrides: Optional[Dict[str, float | int]] = None) -> StrategyParameters:
    settings = cfg.strategy.default_parameters.copy()
    if cfg.run_mode == "demo-live":
        settings.update(cfg.strategy.demo_live_overrides)
    if overrides:
        settings.update(overrides)
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
    )


__all__ = ["StrategyParameters", "StrategySignal", "EmaRsiAtrStrategy", "build_parameters"]
