from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from core.models import Candle, Side


@dataclass(slots=True)
class ScalpingParams:
    preset: str = "HYPER_AGGRESSIVE"
    long_oversold_k: float = 45.0
    short_overbought_k: float = 55.0
    long_cross_min_k: float = 0.0
    short_cross_max_k: float = 100.0
    min_bars_between_trades: int = 0
    disable_trend_filter: bool = True

    @classmethod
    def aggressive(cls) -> "ScalpingParams":
        return cls(
            preset="AGGRESSIVE",
            long_oversold_k=40.0,
            short_overbought_k=60.0,
            long_cross_min_k=0.0,
            short_cross_max_k=100.0,
            min_bars_between_trades=1,
            disable_trend_filter=False,
        )

    @classmethod
    def hyper_aggressive(cls) -> "ScalpingParams":
        return cls()

    def sanitized(self) -> "ScalpingParams":
        return ScalpingParams(
            preset=(self.preset or "HYPER_AGGRESSIVE").upper(),
            long_oversold_k=max(0.0, min(100.0, self.long_oversold_k)),
            short_overbought_k=max(0.0, min(100.0, self.short_overbought_k)),
            long_cross_min_k=max(0.0, min(100.0, self.long_cross_min_k)),
            short_cross_max_k=max(0.0, min(100.0, self.short_cross_max_k)),
            min_bars_between_trades=max(0, self.min_bars_between_trades),
            disable_trend_filter=bool(self.disable_trend_filter),
        )

    def as_dict(self) -> Dict[str, float | int | str]:
        return {
            "preset": self.preset,
            "long_oversold_k": self.long_oversold_k,
            "short_overbought_k": self.short_overbought_k,
            "long_cross_min_k": self.long_cross_min_k,
            "short_cross_max_k": self.short_cross_max_k,
            "min_bars_between_trades": self.min_bars_between_trades,
            "disable_trend_filter": self.disable_trend_filter,
        }


@dataclass(slots=True)
class ScalpingConfig:
    fast_ema: int = 50
    slow_ema: int = 200
    k_period: int = 14
    d_period: int = 3
    stop_loss_pct: float = 0.002
    take_profit_pct: float = 0.003
    size_usd: float = 1_000.0
    params: ScalpingParams = field(default_factory=ScalpingParams.hyper_aggressive)


IndicatorValue = float | int | bool | str | None


@dataclass(slots=True)
class ScalpingDecision:
    action: str
    size_usd: float
    sl_pct: float
    tp_pct: float
    reason: str = ""
    indicators: Dict[str, IndicatorValue] = field(default_factory=dict)
    position_state: str = "FLAT"


class EMAStochasticStrategy:
    """Aggressive EMA + stochastic scalper with trend filter and stateful exits."""

    def __init__(self, config: ScalpingConfig | None = None) -> None:
        self.config = config or ScalpingConfig()
        self.config.params = (self.config.params or ScalpingParams.hyper_aggressive()).sanitized()
        self.params = self.config.params
        self._last_snapshot: Optional[Dict[str, IndicatorValue]] = None
        self._last_trade_index: Optional[int] = None

    @property
    def latest_snapshot(self) -> Optional[Dict[str, IndicatorValue]]:
        return self._last_snapshot

    def evaluate(self, candles: Sequence[Candle], position_side: Side | str | None = None) -> ScalpingDecision:
        position_state = self._normalize_position_state(position_side)
        required = max(self.config.slow_ema, self.config.k_period)
        if len(candles) < required:
            self._last_snapshot = None
            return self._flat(reason="insufficient_data", position_state=position_state)

        closes = [float(c.close) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        price = closes[-1]

        ema_fast = self._ema_latest(closes, self.config.fast_ema)
        ema_slow = self._ema_latest(closes, self.config.slow_ema)
        if ema_fast is None or ema_slow is None:
            self._last_snapshot = None
            return self._flat(reason="ema_warmup", position_state=position_state)

        idx_now = len(candles) - 1
        idx_prev = len(candles) - 2
        k_now = self._stoch_k(highs, lows, closes, idx_now)
        k_prev = self._stoch_k(highs, lows, closes, idx_prev)
        if k_now is None or k_prev is None:
            self._last_snapshot = None
            return self._flat(reason="stoch_warmup", position_state=position_state)

        d_now = self._stoch_d(highs, lows, closes, idx_now)
        d_prev = self._stoch_d(highs, lows, closes, idx_prev)
        if d_now is None or d_prev is None:
            self._last_snapshot = None
            return self._flat(reason="stoch_warmup", position_state=position_state)

        trend_filter_active = not self.params.disable_trend_filter
        long_trend = (ema_fast > ema_slow) if trend_filter_active else True
        short_trend = (ema_fast < ema_slow) if trend_filter_active else True

        params = self.params
        long_oversold = k_now is not None and k_now <= params.long_oversold_k
        short_overbought = k_now is not None and k_now >= params.short_overbought_k

        long_cross = (
            long_trend
            and k_prev is not None
            and d_prev is not None
            and k_now is not None
            and d_now is not None
            and k_prev < d_prev
            and k_now > d_now
            and k_now >= params.long_cross_min_k
            and k_now <= params.short_overbought_k
        )
        short_cross = (
            short_trend
            and k_prev is not None
            and d_prev is not None
            and k_now is not None
            and d_now is not None
            and k_prev > d_prev
            and k_now < d_now
            and k_now <= params.short_cross_max_k
            and k_now >= params.long_oversold_k
        )

        long_signal = long_trend and (long_oversold or long_cross)
        short_signal = short_trend and (short_overbought or short_cross)

        if long_cross:
            long_reason = "long_trend_stoch_cross"
        elif long_oversold:
            long_reason = "long_trend_oversold"
        else:
            long_reason = "long_signal"

        if short_cross:
            short_reason = "short_trend_stoch_cross"
        elif short_overbought:
            short_reason = "short_trend_overbought"
        else:
            short_reason = "short_signal"

        base_indicators = {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "stoch_k": k_now,
            "stoch_d": d_now,
            "stoch_k_prev": k_prev,
            "stoch_d_prev": d_prev,
            "params_preset": params.preset,
            "trend_filter_active": trend_filter_active,
        }

        self._last_snapshot = {
            "price": price,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "stoch_k": k_now,
            "stoch_k_prev": k_prev,
            "stoch_d": d_now,
            "stoch_d_prev": d_prev,
            "position_state": position_state,
            "params_preset": params.preset,
            "last_trade_bar": self._last_trade_index,
            "trend_filter_active": trend_filter_active,
            "throttle_reason": None,
        }

        def _mark_throttle(value: Optional[str]) -> None:
            if self._last_snapshot is not None:
                self._last_snapshot["throttle_reason"] = value

        if position_state == Side.FLAT.value:
            if long_signal and self._allow_new_trade(idx_now):
                return self._enter_trade("LONG", long_reason, base_indicators, position_state, idx_now)
            if short_signal and self._allow_new_trade(idx_now):
                return self._enter_trade("SHORT", short_reason, base_indicators, position_state, idx_now)
            throttle_reason: Optional[str] = None
            if trend_filter_active and not (long_trend or short_trend):
                throttle_reason = "trend_filter"
            elif (long_signal or short_signal) and not self._allow_new_trade(idx_now):
                throttle_reason = "cooldown"
            _mark_throttle(throttle_reason)
            return self._flat(
                reason=throttle_reason or "no_signal",
                indicators=base_indicators,
                position_state=position_state,
            )

        if position_state == Side.LONG.value:
            if short_signal:
                return self._close_position(
                    "CLOSE_LONG",
                    "close_long_opposite_signal",
                    base_indicators,
                    position_state,
                )
            return self._hold_position(base_indicators, position_state, reason="hold_position_no_new_signal")

        if position_state == Side.SHORT.value:
            if long_signal:
                return self._close_position(
                    "CLOSE_SHORT",
                    "close_short_opposite_signal",
                    base_indicators,
                    position_state,
                )
            return self._hold_position(base_indicators, position_state, reason="hold_position_no_new_signal")

        return self._flat(reason="no_signal", indicators=base_indicators, position_state=position_state)

    def _enter_trade(
        self,
        action: str,
        reason: str,
        indicators: Dict[str, IndicatorValue],
        position_state: str,
        bar_index: int,
    ) -> ScalpingDecision:
        self._last_trade_index = bar_index
        decision = ScalpingDecision(
            action=action,
            size_usd=self.config.size_usd,
            sl_pct=self.config.stop_loss_pct,
            tp_pct=self.config.take_profit_pct,
            reason=reason,
            indicators=indicators,
            position_state=position_state,
        )
        self._update_snapshot(
            last_action=action,
            reason=reason,
            position_state=position_state,
            throttle_reason=None,
            indicators=indicators,
        )
        return decision

    def _close_position(
        self,
        action: str,
        reason: str,
        indicators: Dict[str, IndicatorValue],
        position_state: str,
    ) -> ScalpingDecision:
        decision = ScalpingDecision(
            action=action,
            size_usd=0.0,
            sl_pct=self.config.stop_loss_pct,
            tp_pct=self.config.take_profit_pct,
            reason=reason,
            indicators=indicators,
            position_state=position_state,
        )
        self._update_snapshot(
            last_action=action,
            reason=reason,
            position_state=position_state,
            throttle_reason=None,
            indicators=indicators,
        )
        return decision

    def _hold_position(
        self,
        indicators: Dict[str, IndicatorValue],
        position_state: str,
        *,
        reason: str,
    ) -> ScalpingDecision:
        decision = ScalpingDecision(
            action="HOLD",
            size_usd=0.0,
            sl_pct=self.config.stop_loss_pct,
            tp_pct=self.config.take_profit_pct,
            reason=reason,
            indicators=indicators,
            position_state=position_state,
        )
        throttle_reason = (self._last_snapshot or {}).get("throttle_reason")
        self._update_snapshot(
            last_action="HOLD",
            reason=reason,
            position_state=position_state,
            throttle_reason=throttle_reason,
            indicators=indicators,
        )
        return decision

    def _update_snapshot(self, **fields: IndicatorValue) -> None:
        snapshot = dict(self._last_snapshot or {})
        snapshot.update(fields)
        self._last_snapshot = snapshot

    def _flat(
        self,
        *,
        reason: str,
        indicators: Optional[Dict[str, IndicatorValue]] = None,
        position_state: str,
    ) -> ScalpingDecision:
        return ScalpingDecision(
            action="FLAT",
            size_usd=0.0,
            sl_pct=self.config.stop_loss_pct,
            tp_pct=self.config.take_profit_pct,
            reason=reason,
            indicators=indicators or {},
            position_state=position_state,
        )

    def _allow_new_trade(self, bar_index: int) -> bool:
        if self.params.min_bars_between_trades <= 0:
            return True
        if self._last_trade_index is None:
            return True
        return (bar_index - self._last_trade_index) >= self.params.min_bars_between_trades

    def _normalize_position_state(self, position_side: Side | str | None) -> str:
        if isinstance(position_side, Side):
            return position_side.value
        if isinstance(position_side, str):
            upper = position_side.upper()
            if upper in {Side.LONG.value, Side.SHORT.value, Side.FLAT.value}:
                return upper
        return Side.FLAT.value

    def _ema_latest(self, values: Sequence[float], length: int) -> Optional[float]:
        if length <= 0 or len(values) < length:
            return None
        multiplier = 2 / (length + 1)
        ema = sum(values[:length]) / length
        for price in values[length:]:
            ema += (price - ema) * multiplier
        return float(ema)

    def _stoch_k(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], index: int) -> Optional[float]:
        start = index - self.config.k_period + 1
        if start < 0:
            return None
        window_high = max(highs[start : index + 1])
        window_low = min(lows[start : index + 1])
        span = window_high - window_low
        if span <= 0:
            return None
        return float((closes[index] - window_low) / span * 100)

    def _stoch_d(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], index: int) -> Optional[float]:
        if self.config.d_period <= 1:
            return self._stoch_k(highs, lows, closes, index)
        values: list[float] = []
        for offset in range(self.config.d_period):
            sample_idx = index - offset
            if sample_idx < 0:
                return None
            k_value = self._stoch_k(highs, lows, closes, sample_idx)
            if k_value is None:
                return None
            values.append(k_value)
        if len(values) < self.config.d_period:
            return None
        return float(sum(values) / len(values))
