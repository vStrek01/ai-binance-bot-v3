"""Lightweight learning loop that adapts parameters from live trades."""
# EXPERIMENTAL — DO NOT ENABLE FOR LIVE TRADING WITHOUT SEPARATE VALIDATION.
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from bot.core.config import BotConfig, ensure_directories
from bot.strategies import StrategyParameters
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


class TradeLearningStore:
    def __init__(
        self,
        cfg: BotConfig,
        path: Path | None = None,
        history_path: Path | None = None,
        window: int | None = None,
    ) -> None:
        ensure_directories(cfg.paths, extra=[cfg.paths.optimization_dir, cfg.paths.log_dir])
        self._config = cfg
        self.path = path or (cfg.paths.optimization_dir / "learned_params.json")
        self.history_path = history_path or (cfg.paths.log_dir / "learned_trades.jsonl")
        self.window = window or cfg.runtime.learning_window
        self._state: Dict[str, Dict[str, Dict[str, Any]]] = self._load_state()

    def _load_state(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            logger.warning("learned_params.json corrupted; resetting state")
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload  # type: ignore[return-value]

    def _persist(self) -> None:
        atomic_write_text(self.path, json.dumps(self._state, indent=2))

    def _append_history(self, record: Dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def record_trade(self, trade_payload: Dict[str, Any], params: StrategyParameters) -> None:
        symbol = str(trade_payload.get("symbol"))
        timeframe = str(trade_payload.get("timeframe"))
        if not symbol or not timeframe:
            return
        market_state = self._state.setdefault(symbol, {}).setdefault(timeframe, {"trades": []})
        trades: List[Dict[str, Any]] = market_state.setdefault("trades", [])
        record = {
            "timestamp": trade_payload.get("closed_at") or trade_payload.get("opened_at"),
            "pnl": _safe_float(trade_payload.get("pnl", 0.0)),
            "mae": _safe_float(trade_payload.get("mae", 0.0)),
            "mfe": _safe_float(trade_payload.get("mfe", 0.0)),
            "side": trade_payload.get("side"),
            "indicators": trade_payload.get("indicators", {}),
        }
        trades.append(record)
        if len(trades) > self.window:
            del trades[0 : len(trades) - self.window]
        stats = self._compute_stats(trades)
        regime = self._detect_regime(trades)
        reward = self._compute_reward(stats, market_state)
        streak = self._update_streak(market_state.get("streak", 0), record["pnl"])
        market_state["stats"] = stats
        market_state["regime"] = regime
        market_state["reward"] = reward
        market_state["streak"] = streak
        tuned_params = self._tune_parameters(asdict(params), stats, market_state)
        market_state["params"] = tuned_params
        market_state["updated_at"] = time.time()
        self._persist()
        history_record = {
            **trade_payload,
            "learned_params": tuned_params,
            "stats": stats,
        }
        self._append_history(history_record)

    def _compute_stats(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        if not trades:
            return {
                "avg_pnl": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_mae": 0.0,
                "avg_mfe": 0.0,
                "trades": 0.0,
            }
        pnls = [_safe_float(t.get("pnl", 0.0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        if losses and (loss_sum := sum(losses)) != 0:
            profit_factor = sum(wins) / abs(loss_sum)
        else:
            profit_factor = float("inf") if wins else 0.0
        if not math.isfinite(profit_factor):
            profit_factor = 10.0 if wins else 0.0
        avg_mae = statistics.mean(_safe_float(t.get("mae", 0.0)) for t in trades)
        avg_mfe = statistics.mean(_safe_float(t.get("mfe", 0.0)) for t in trades)
        return {
            "avg_pnl": statistics.mean(pnls),
            "win_rate": len(wins) / len(pnls),
            "profit_factor": profit_factor,
            "avg_mae": avg_mae,
            "avg_mfe": avg_mfe,
            "trades": float(len(trades)),
        }

    def _tune_parameters(self, params: Dict[str, Any], stats: Dict[str, float], market_state: Dict[str, Any]) -> Dict[str, Any]:
        tuned = dict(params)
        avg_pnl = stats.get("avg_pnl", 0.0)
        win_rate = stats.get("win_rate", 0.0)
        profit_factor = stats.get("profit_factor", 0.0)
        if avg_pnl < 0:
            tuned["atr_stop"] = min(float(tuned.get("atr_stop", 2.0)) + 0.1, 4.0)
            tuned["atr_target"] = max(float(tuned.get("atr_target", 3.0)) - 0.1, 1.5)
        elif profit_factor > 1.5:
            tuned["atr_target"] = min(float(tuned.get("atr_target", 3.0)) + 0.15, 5.0)
        if win_rate < 0.45:
            tuned["rsi_overbought"] = max(float(tuned.get("rsi_overbought", 65)) - 1, 55)
            tuned["rsi_oversold"] = min(float(tuned.get("rsi_oversold", 35)) + 1, 45)
        elif win_rate > 0.6:
            tuned["rsi_overbought"] = min(float(tuned.get("rsi_overbought", 65)) + 1, 80)
            tuned["rsi_oversold"] = max(float(tuned.get("rsi_oversold", 35)) - 1, 20)
        tuned["fast_ema"] = int(max(5, min(100, round(float(tuned.get("fast_ema", 21))))))
        tuned["slow_ema"] = int(max(20, min(200, round(float(tuned.get("slow_ema", 55))))))
        tuned["atr_stop"] = float(tuned.get("atr_stop", 2.0))
        tuned["atr_target"] = float(tuned.get("atr_target", 3.0))
        tuned["rsi_overbought"] = float(tuned.get("rsi_overbought", 65))
        tuned["rsi_oversold"] = float(tuned.get("rsi_oversold", 35))
        tuned = self._reinforcement_adjust(tuned, stats, market_state)
        return tuned

    def _detect_regime(self, trades: List[Dict[str, Any]]) -> str:
        if len(trades) < 5:
            return "neutral"
        pnls = [_safe_float(t.get("pnl", 0.0)) for t in trades]
        stdev = statistics.pstdev(pnls) if len(pnls) > 1 else 0.0
        avg = statistics.mean(pnls)
        if stdev > self._config.reinforcement.volatility_threshold:
            return "volatile_up" if avg > 0 else "volatile_down"
        if avg > 0:
            return "trend_up"
        if avg < 0:
            return "trend_down"
        return "neutral"

    def _compute_reward(self, stats: Dict[str, float], market_state: Dict[str, Any]) -> float:
        cfg = self._config.reinforcement
        avg_pnl = stats.get("avg_pnl", 0.0)
        mae = abs(stats.get("avg_mae", 1.0)) or 1.0
        norm = avg_pnl / mae
        profit_factor = stats.get("profit_factor", 0.0)
        win_rate = stats.get("win_rate", 0.0)
        reward = norm * cfg.reward_alpha + profit_factor * (1 - cfg.reward_alpha)
        trade_density = stats.get("trades", 0) / max(cfg.regime_window, 1)
        penalty = 0.0
        if trade_density > 1.0:
            penalty += cfg.penalty_alpha * (trade_density - 1.0)
        streak = market_state.get("streak", 0)
        if streak <= -cfg.loss_streak_threshold:
            penalty += cfg.penalty_alpha * abs(streak)
        return reward - penalty

    def _update_streak(self, streak: int, pnl: float) -> int:
        if pnl > 0:
            return streak + 1 if streak >= 0 else 1
        if pnl < 0:
            return streak - 1 if streak <= 0 else -1
        return streak

    def best_params(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        payload = self._state.get(symbol, {}).get(timeframe)
        if not payload:
            return None
        params = payload.get("params")
        if not isinstance(params, dict):
            return None
        return dict(params)

    def trade_count(self, symbol: str, timeframe: str) -> int:
        payload = self._state.get(symbol, {}).get(timeframe)
        if not payload:
            return 0
        trades = payload.get("trades", [])
        if not isinstance(trades, list):
            return 0
        return len(trades)

    def _reinforcement_adjust(
        self,
        params: Dict[str, Any],
        stats: Dict[str, float],
        market_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg = self._config.reinforcement
        reward = market_state.get("reward", 0.0)
        regime = market_state.get("regime", "neutral")
        streak = market_state.get("streak", 0)
        updated = dict(params)
        defaults = self._config.strategy.default_parameters
        def clamp(key: str, value: float, minimum: float, maximum: float) -> float:
            return max(minimum, min(maximum, value))

        if reward > 0.5:
            updated["atr_target"] = clamp("atr_target", updated.get("atr_target", 3.0) + 0.1, 1.0, 6.0)
            updated["cooldown_bars"] = int(clamp("cooldown_bars", updated.get("cooldown_bars", 2) - 1, 0, 10))
        elif reward < -0.5:
            updated = self._blend_with_defaults(updated, defaults, blend=0.35)
            updated["cooldown_bars"] = int(clamp("cooldown_bars", updated.get("cooldown_bars", 2) + 1, 0, 20))

        if "volatile" in regime:
            updated["atr_stop"] = clamp("atr_stop", updated.get("atr_stop", 1.6) + 0.2, 1.0, 5.0)
        elif regime == "trend_up":
            updated["fast_ema"] = int(clamp("fast_ema", updated.get("fast_ema", 13) - 1, 5, 50))
        elif regime == "trend_down":
            updated["slow_ema"] = int(clamp("slow_ema", updated.get("slow_ema", 34) + 2, 20, 200))

        if streak <= -cfg.loss_streak_threshold:
            updated = self._blend_with_defaults(updated, defaults, blend=0.5)
            updated["rsi_overbought"] = clamp("rsi_overbought", updated.get("rsi_overbought", 60) - 1, 50, 80)
            updated["rsi_oversold"] = clamp("rsi_oversold", updated.get("rsi_oversold", 40) + 1, 20, 60)
        if streak >= cfg.win_streak_threshold:
            updated["hold_bars"] = int(clamp("hold_bars", updated.get("hold_bars", 90) + 5, 30, 300))

        trade_volume = market_state.get("stats", {}).get("trades", 0)
        if trade_volume and trade_volume > cfg.regime_window:
            updated["cooldown_bars"] = int(clamp("cooldown_bars", updated.get("cooldown_bars", 2) + 1, 0, 30))
        return updated

    @staticmethod
    def _blend_with_defaults(params: Dict[str, Any], defaults: Dict[str, float], blend: float) -> Dict[str, Any]:
        updated: Dict[str, Any] = dict(params)
        for key, value in defaults.items():
            current = _safe_float(updated.get(key, value), float(value))
            updated[key] = current * (1 - blend) + float(value) * blend
        return updated


__all__ = ["TradeLearningStore"]
