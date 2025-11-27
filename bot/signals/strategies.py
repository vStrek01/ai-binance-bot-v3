# Goal: high-frequency EMA/RSI/ATR strategy for 1m futures, frequent LONG/SHORT signals on majors, reusing existing indicators + risk stack.
# Indicators: ema_fast, ema_slow, rsi, atr, atr_pct, spread_pct, trend_ema derived from bot.signals.indicators helpers.
# Entry logic: EMA trend alignment plus RSI hover near 50 to catch pullbacks; ATR multipliers drive stop/target bands.
# Guards: volatility floor, spread cap, optional trend veto, RSI warmup skip, spacing model plus cooldown telemetry snapshotting.
# generate_signals must emit StrategySignal objects for downstream execution/risk engines without altering external interfaces.
"""Strategy definitions shared across backtest, execution, and RL."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from bot.core.config import BotConfig
from bot.signals import indicators
from infra.logging import log_event


_INTERVAL_TO_MINUTES: Dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
}


def _coerce_time(value: Any) -> Optional[time]:
    if isinstance(value, time):
        return value
    if value is None:
        return None
    try:
        timestamp = pd.to_datetime(value, errors="coerce")
    except Exception:  # noqa: BLE001 - defensive parsing
        return None
    if pd.isna(timestamp):
        return None
    dt_value = timestamp.to_pydatetime()
    return dt_value.time().replace(tzinfo=None)


def _normalize_session_windows(windows: Iterable[Any]) -> List[Tuple[time, time]]:
    normalized: List[Tuple[time, time]] = []
    for window in windows or []:
        start = getattr(window, "start", None)
        end = getattr(window, "end", None)
        if isinstance(window, dict):
            start = start or window.get("start")
            end = end or window.get("end")
        start_time = _coerce_time(start)
        end_time = _coerce_time(end)
        if start_time and end_time:
            normalized.append((start_time, end_time))
    return normalized


def _interval_ratio(base_interval: str, higher_interval: Optional[str]) -> int:
    if not higher_interval:
        return 0
    base_minutes = _INTERVAL_TO_MINUTES.get(base_interval.lower(), 1)
    higher_minutes = _INTERVAL_TO_MINUTES.get(higher_interval.lower())
    if higher_minutes is None or higher_minutes <= base_minutes:
        return 0
    return max(1, higher_minutes // base_minutes)


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
    min_volatility_pct: float = 0.01
    max_spread_pct: float = 0.06
    max_volatility_pct: float = 0.0
    trend_confirm_bars: int = 0
    min_reentry_bars_same_dir: int = 15
    min_reentry_bars_flip: int = 5
    higher_tf_trend_bars: int = 0
    trading_timezone: str = "UTC"
    session_windows: List[Tuple[time, time]] = field(default_factory=list)


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
        self._session_windows = list(params.session_windows)
        self._timezone = params.trading_timezone or "UTC"

    def _annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy().reset_index(drop=True)
        if enriched.empty:
            return enriched

        close = enriched["close"]
        high = enriched["high"]
        low = enriched["low"]
        close_safe = close.replace(0, pd.NA)

        enriched["ema_fast"] = indicators.ema(close, self.params.fast_ema)
        enriched["ema_slow"] = indicators.ema(close, self.params.slow_ema)
        enriched["rsi"] = indicators.rsi(close, self.params.rsi_length)
        enriched["atr"] = indicators.atr(enriched, self.params.atr_period)
        enriched["atr_pct"] = (enriched["atr"] / close_safe * 100).fillna(0.0)
        enriched["spread_pct"] = ((high - low).abs() / close_safe * 100).fillna(0.0)

        trend_window = int(self.params.trend_confirm_bars or 0)
        if trend_window <= 0:
            trend_window = int(self.params.slow_ema)

        if trend_window > 0:
            enriched["trend_ema"] = indicators.ema(close, trend_window)
        else:
            enriched["trend_ema"] = pd.Series(pd.NA, index=enriched.index, dtype="float64")

        higher_tf_window = int(self.params.higher_tf_trend_bars or 0)
        if higher_tf_window > 0:
            enriched["higher_tf_trend"] = indicators.ema(close, higher_tf_window)
        else:
            enriched["higher_tf_trend"] = pd.Series(pd.NA, index=enriched.index, dtype="float64")
        return enriched

    def generate_signals(self, frame: pd.DataFrame) -> List[StrategySignal]:
        annotated = self._annotate(frame)
        signals: List[StrategySignal] = []
        if annotated.empty:
            return signals

        cooldown = 0
        total_rows = len(annotated)
        last_signal_idx = -1_000_000_000
        last_signal_dir = 0

        for idx, row in enumerate(annotated.itertuples(), start=0):
            last_bar = idx == total_rows - 1

            atr_pct_row = float(getattr(row, "atr_pct", 0.0) or 0.0)
            spread_pct_row = float(getattr(row, "spread_pct", 0.0) or 0.0)
            trend_value = getattr(row, "trend_ema", None)
            higher_tf_value = getattr(row, "higher_tf_trend", None)
            timestamp_value = getattr(row, "close_time", None) or getattr(row, "open_time", None)

            if self._session_windows and self._in_session_blackout(timestamp_value):
                if last_bar:
                    self._log_veto("session", "blackout_window")
                    self._record_snapshot(row, cooldown, atr_pct=atr_pct_row, spread_pct=spread_pct_row, trend=trend_value)
                continue

            if cooldown > 0:
                remaining = cooldown
                cooldown -= 1
                if last_bar:
                    self._log_veto("cooldown", "cooldown_active", bars_remaining=remaining)
                    self._record_snapshot(row, remaining, atr_pct=atr_pct_row, spread_pct=spread_pct_row, trend=trend_value)
                continue

            required_values = [row.atr, row.ema_fast, row.ema_slow, row.rsi]
            if any(pd.isna(val) for val in required_values):
                continue

            price = float(row.close)
            ema_fast = float(row.ema_fast)
            ema_slow = float(row.ema_slow)
            rsi_value = float(row.rsi)
            atr_value = float(row.atr)

            if rsi_value <= 0.0 or rsi_value >= 100.0:
                continue

            effective_min_vol = max(self.params.min_volatility_pct, 0.0)
            if effective_min_vol > 0 and atr_pct_row < effective_min_vol:
                if last_bar:
                    self._log_veto("volatility", "below_threshold", atr_pct=atr_pct_row, threshold=effective_min_vol)
                    self._record_snapshot(row, cooldown, atr_pct=atr_pct_row, spread_pct=spread_pct_row, trend=trend_value)
                continue

            effective_max_vol = max(self.params.max_volatility_pct, 0.0)
            if effective_max_vol > 0 and atr_pct_row > effective_max_vol:
                if last_bar:
                    self._log_veto("volatility", "above_threshold", atr_pct=atr_pct_row, threshold=effective_max_vol)
                    self._record_snapshot(row, cooldown, atr_pct=atr_pct_row, spread_pct=spread_pct_row, trend=trend_value)
                continue

            hard_spread_limit = max(self.params.max_spread_pct, 0.0)
            if 0 < hard_spread_limit <= 1.0:
                hard_spread_limit *= 100.0  # allow config values expressed as proportions (0.05 == 5%)
            if hard_spread_limit > 0 and spread_pct_row > hard_spread_limit:
                if last_bar:
                    self._log_veto("spread", "exceeds_threshold", spread_pct=spread_pct_row, threshold=hard_spread_limit)
                    self._record_snapshot(row, cooldown, atr_pct=atr_pct_row, spread_pct=spread_pct_row, trend=trend_value)
                continue

            trend_up = ema_fast >= ema_slow
            trend_down = ema_fast <= ema_slow
            rsi_mid = 50.0
            rsi_long_trigger = min(self.params.rsi_oversold + 5.0, rsi_mid)
            rsi_short_trigger = max(self.params.rsi_overbought - 5.0, rsi_mid)

            long_cond = trend_up and rsi_value <= rsi_long_trigger
            short_cond = trend_down and rsi_value >= rsi_short_trigger

            candidate_dir = 0
            if long_cond and not short_cond:
                candidate_dir = 1
            elif short_cond and not long_cond:
                candidate_dir = -1

            if candidate_dir == 0:
                if last_bar:
                    self._record_snapshot(row, cooldown, atr_pct=atr_pct_row, spread_pct=spread_pct_row, trend=trend_value)
                continue

            bars_since_last = idx - last_signal_idx
            same_reentry = max(int(self.params.min_reentry_bars_same_dir), 0)
            flip_reentry = max(int(self.params.min_reentry_bars_flip), 0)
            if last_signal_dir == 0:
                allow = True
            elif candidate_dir == last_signal_dir:
                allow = bars_since_last >= same_reentry
            else:
                allow = bars_since_last >= flip_reentry

            if not allow:
                continue

            direction = candidate_dir
            if direction == 1:
                stop_loss = price - atr_value * self.params.atr_stop
                take_profit = price + atr_value * self.params.atr_target
            else:
                stop_loss = price + atr_value * self.params.atr_stop
                take_profit = price - atr_value * self.params.atr_target

            if not self._trend_confirmed(direction, price, trend_value):
                direction_label = "LONG" if direction == 1 else "SHORT"
                trend_float = None
                try:
                    trend_float = float(trend_value)
                except (TypeError, ValueError):
                    trend_float = None
                else:
                    if pd.isna(trend_float):
                        trend_float = None
                self._log_veto("trend", "misaligned", trend=trend_float, price=price, direction=direction_label)
                self._record_snapshot(row, cooldown, atr_pct=atr_pct_row, spread_pct=spread_pct_row, trend=trend_float)
                continue

            if self.params.higher_tf_trend_bars > 0 and not self._higher_tf_confirmed(direction, price, higher_tf_value):
                direction_label = "LONG" if direction == 1 else "SHORT"
                higher_float = None
                try:
                    higher_float = float(higher_tf_value)
                except (TypeError, ValueError):
                    higher_float = None
                else:
                    if pd.isna(higher_float):
                        higher_float = None
                self._log_veto("higher_tf", "misaligned", trend=higher_float, price=price, direction=direction_label)
                self._record_snapshot(row, cooldown, atr_pct=atr_pct_row, spread_pct=spread_pct_row, trend=higher_float)
                continue

            signals.append(
                StrategySignal(
                    index=idx,
                    direction=direction,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    indicators={
                        "ema_fast": ema_fast,
                        "ema_slow": ema_slow,
                        "rsi": rsi_value,
                        "atr": atr_value,
                        "atr_pct": atr_pct_row,
                        "spread_pct": spread_pct_row,
                    },
                    reason="ema_rsi_trend_pullback",
                )
            )

            last_signal_idx = idx
            last_signal_dir = direction
            cooldown = max(int(self.params.cooldown_bars), 0)
            if last_bar:
                self._record_snapshot(row, cooldown, atr_pct=atr_pct_row, spread_pct=spread_pct_row, trend=trend_value)

        return signals

    def latest_snapshot(self) -> Optional[Dict[str, float]]:
        return self._last_snapshot

    def _record_snapshot(self, row: pd.Series | pd.Index, cooldown: int, **extras: Any) -> None:
        def _safe_float(value: Any) -> Optional[float]:
            try:
                result = float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return None
            return None if pd.isna(result) else result

        price = _safe_float(getattr(row, "close", None))
        snapshot = {
            "last_close": price,
            "ema_fast": _safe_float(getattr(row, "ema_fast", None)),
            "ema_slow": _safe_float(getattr(row, "ema_slow", None)),
            "rsi": _safe_float(getattr(row, "rsi", None)),
            "atr": _safe_float(getattr(row, "atr", None)),
            "cooldown_bars": int(max(cooldown, 0)),
            "atr_pct": _safe_float(getattr(row, "atr_pct", None)) or 0.0,
            "spread_pct": _safe_float(getattr(row, "spread_pct", None)) or 0.0,
            "trend": _safe_float(getattr(row, "trend_ema", None)),
            "higher_tf_trend": _safe_float(getattr(row, "higher_tf_trend", None)),
        }
        snapshot.update({k: v for k, v in extras.items()})
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
        if pd.isna(trend) or trend <= 0:
            return True

        distance_pct = (price - trend) / trend * 100
        tolerance = 0.25
        if direction == 1:
            return distance_pct >= -tolerance
        if direction == -1:
            return distance_pct <= tolerance
        return True

    def _higher_tf_confirmed(self, direction: int, price: float, trend_value: Any) -> bool:
        try:
            trend = float(trend_value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return True
        if pd.isna(trend) or trend <= 0:
            return True
        if direction == 1:
            return price >= trend
        if direction == -1:
            return price <= trend
        return True

    def _normalize_timestamp(self, value: Any) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        try:
            timestamp = pd.to_datetime(value, utc=True, errors="coerce")
        except Exception:  # noqa: BLE001 - defensive
            return None
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp

    def _in_session_blackout(self, value: Any) -> bool:
        if not self._session_windows:
            return False
        normalized = self._normalize_timestamp(value)
        if normalized is None:
            return False
        try:
            localized = normalized.tz_convert(self._timezone)
        except Exception:  # noqa: BLE001 - fall back to UTC
            localized = normalized
        current_time = localized.to_pydatetime().time().replace(tzinfo=None)
        for start, end in self._session_windows:
            if start < end:
                if start <= current_time < end:
                    return True
            else:  # window spans midnight
                if current_time >= start or current_time < end:
                    return True
        return False


def build_parameters(
    cfg: BotConfig,
    symbol: Optional[str] = None,
    overrides: Optional[Dict[str, float | int]] = None,
) -> StrategyParameters:
    settings: Dict[str, float | int] = dict(cfg.strategy.default_parameters)
    if cfg.run_mode in {"demo-live", "live"}:
        demo_live_overrides = getattr(cfg.strategy, "demo_live_overrides", None) or {}
        settings.update(demo_live_overrides)
        if symbol:
            symbol_map = getattr(cfg.strategy, "symbol_overrides", None) or {}
            sym_key = symbol.upper()
            for key, value in symbol_map.items():
                if key.upper() == sym_key and value:
                    settings.update(value)
                    break
    if overrides:
        settings.update(overrides)

    baseline = cfg.baseline_strategy
    baseline_min_atr = float(getattr(baseline, "min_atr_pct", 0.0) or 0.0)
    baseline_max_atr = float(getattr(baseline, "max_atr_pct", 0.0) or 0.0)
    baseline_slow_ema = int(getattr(baseline, "slow_ema", 0) or 0)
    higher_tf_interval = getattr(baseline, "higher_tf_trend_interval", None)
    higher_tf_ratio = _interval_ratio(cfg.interval, higher_tf_interval)
    session_windows = _normalize_session_windows(getattr(baseline, "no_trade_sessions", []))

    if cfg.run_mode in {"demo-live", "live"}:
        default_min_vol = float(settings.get("min_volatility_pct", baseline_min_atr * 0.25))
        default_trend_bars = int(settings.get("trend_confirm_bars", higher_tf_ratio * baseline_slow_ema))
    else:
        default_min_vol = float(settings.get("min_volatility_pct", baseline_min_atr))
        default_trend_bars = int(settings.get("trend_confirm_bars", higher_tf_ratio * baseline_slow_ema or baseline_slow_ema * 2))

    min_volatility_pct = float(settings.get("min_volatility_pct", default_min_vol))
    max_spread_pct = float(settings.get("max_spread_pct", 0.06))
    trend_confirm_bars = int(settings.get("trend_confirm_bars", default_trend_bars))
    min_reentry_same = int(settings.get("min_reentry_bars_same_dir", 15))
    min_reentry_flip = int(settings.get("min_reentry_bars_flip", 5))
    max_volatility_pct = float(settings.get("max_volatility_pct", baseline_max_atr))
    higher_tf_trend_bars = int(settings.get("higher_tf_trend_bars", higher_tf_ratio * baseline_slow_ema)) if higher_tf_ratio else int(settings.get("higher_tf_trend_bars", 0))

    return StrategyParameters(
        fast_ema=int(settings["fast_ema"]),
        slow_ema=int(settings["slow_ema"]),
        rsi_length=int(settings.get("rsi_length", 14)),
        rsi_overbought=float(settings["rsi_overbought"]),
        rsi_oversold=float(settings["rsi_oversold"]),
        atr_period=int(settings.get("atr_period", 14)),
        atr_stop=float(settings["atr_stop"]),
        atr_target=float(settings["atr_target"]),
        cooldown_bars=int(settings.get("cooldown_bars", 1)),
        hold_bars=int(settings.get("hold_bars", 45)),
        min_volatility_pct=min_volatility_pct,
        max_spread_pct=max_spread_pct,
        max_volatility_pct=max_volatility_pct,
        trend_confirm_bars=trend_confirm_bars,
        min_reentry_bars_same_dir=min_reentry_same,
        min_reentry_bars_flip=min_reentry_flip,
        higher_tf_trend_bars=higher_tf_trend_bars,
        trading_timezone=getattr(cfg, "trading_timezone", "UTC") or "UTC",
        session_windows=session_windows,
    )


__all__ = ["StrategyParameters", "StrategySignal", "EmaRsiAtrStrategy", "build_parameters"]
