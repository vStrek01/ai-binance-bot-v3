from __future__ import annotations

import os
from typing import Optional, Tuple

from bot.core.config import BotConfig
from strategies.ema_stoch_scalping import ScalpingConfig, ScalpingParams

StrategyModeLiteral = str

LLM_MODE = "llm"
BASELINE_MODE = "baseline"
SCALPING_MODE = "scalping"

_STRATEGY_ALIASES = {
    "scalping": SCALPING_MODE,
    "ema_stochastic": SCALPING_MODE,
    "ema_stoch": SCALPING_MODE,
    "ema_stoch_scalping": SCALPING_MODE,
    "baseline": BASELINE_MODE,
    "llm": LLM_MODE,
}

_ALLOWED_MODES = {LLM_MODE, BASELINE_MODE, SCALPING_MODE}


def _normalize(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_")
    mapped = _STRATEGY_ALIASES.get(normalized, normalized)
    return mapped if mapped in _ALLOWED_MODES else None


def resolve_strategy_mode(cfg: BotConfig, override: Optional[str] = None) -> str:
    candidates = [override, getattr(cfg, "strategy_mode", None), getattr(cfg, "core_strategy_mode", None)]
    for candidate in candidates:
        normalized = _normalize(candidate)
        if normalized:
            return normalized
    return SCALPING_MODE


def apply_strategy_mode_override(cfg: BotConfig, mode: str) -> None:
    canonical = _normalize(mode) or resolve_strategy_mode(cfg)
    setattr(cfg, "strategy_mode", canonical)
    setattr(cfg, "core_strategy_mode", canonical)


def is_scalping_mode(mode: str) -> bool:
    return (_normalize(mode) or mode) == SCALPING_MODE


def build_scalping_config(cfg: BotConfig) -> ScalpingConfig:
    payload = getattr(cfg, "scalping_strategy", None)
    preset_raw = os.getenv("SCALPING_PRESET") or getattr(payload, "preset", None)
    preset = str(preset_raw or "HYPER_AGGRESSIVE").upper()
    base_params = ScalpingParams.aggressive() if preset == "AGGRESSIVE" else ScalpingParams.hyper_aggressive()
    params = ScalpingParams(
        preset=preset,
        long_oversold_k=float(
            getattr(payload, "long_oversold_k", base_params.long_oversold_k)
            if payload is not None
            else base_params.long_oversold_k
        ),
        short_overbought_k=float(
            getattr(payload, "short_overbought_k", base_params.short_overbought_k)
            if payload is not None
            else base_params.short_overbought_k
        ),
        long_cross_min_k=float(
            getattr(payload, "long_cross_min_k", base_params.long_cross_min_k)
            if payload is not None
            else base_params.long_cross_min_k
        ),
        short_cross_max_k=float(
            getattr(payload, "short_cross_max_k", base_params.short_cross_max_k)
            if payload is not None
            else base_params.short_cross_max_k
        ),
        min_bars_between_trades=int(
            getattr(payload, "min_bars_between_trades", base_params.min_bars_between_trades)
            if payload is not None
            else base_params.min_bars_between_trades
        ),
        disable_trend_filter=bool(
            getattr(payload, "disable_trend_filter", base_params.disable_trend_filter)
            if payload is not None
            else base_params.disable_trend_filter
        ),
    ).sanitized()
    return ScalpingConfig(
        fast_ema=int(getattr(payload, "fast_ema", 50)),
        slow_ema=int(getattr(payload, "slow_ema", 200)),
        k_period=int(getattr(payload, "k_period", 14)),
        d_period=int(getattr(payload, "d_period", 3)),
        stop_loss_pct=float(getattr(payload, "stop_loss_pct", 0.002)),
        take_profit_pct=float(getattr(payload, "take_profit_pct", 0.003)),
        size_usd=float(getattr(payload, "size_usd", 1_000.0)),
        params=params,
    )


def strategy_mode_choices() -> Tuple[str, ...]:
    return tuple(sorted(_ALLOWED_MODES))
