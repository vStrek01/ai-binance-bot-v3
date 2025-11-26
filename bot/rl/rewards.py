"""Reward shaping functions for the RL trainer."""
# EXPERIMENTAL — DO NOT ENABLE FOR LIVE TRADING WITHOUT SEPARATE VALIDATION.
from __future__ import annotations

from typing import Callable, Dict

RewardFn = Callable[[Dict[str, float]], float]


def pnl_reward(context: Dict[str, float]) -> float:
    atr = max(abs(context.get("atr", 1.0)), 1e-6)
    return context.get("pnl_delta", 0.0) / atr


def risk_adjusted_reward(context: Dict[str, float]) -> float:
    reward = pnl_reward(context)
    drawdown = max(context.get("drawdown", 0.0), 0.0)
    reward -= 0.1 * drawdown
    reward -= 0.05 * abs(context.get("position", 0.0))
    return reward


def smoothness_reward(context: Dict[str, float]) -> float:
    reward = risk_adjusted_reward(context)
    reward -= 0.02 * context.get("action_change", 0.0)
    return reward


_REWARD_MAP: Dict[str, RewardFn] = {
    "pnl": pnl_reward,
    "risk_adjusted": risk_adjusted_reward,
    "smooth": smoothness_reward,
}


def get_reward_function(name: str) -> RewardFn:
    return _REWARD_MAP.get(name, risk_adjusted_reward)


__all__ = ["get_reward_function", "pnl_reward", "risk_adjusted_reward", "smoothness_reward", "RewardFn"]
