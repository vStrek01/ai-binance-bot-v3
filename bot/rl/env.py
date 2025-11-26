"""Reinforcement-learning environment for perpetual futures trading."""
# EXPERIMENTAL — DO NOT ENABLE FOR LIVE TRADING WITHOUT SEPARATE VALIDATION.
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bot.core.config import BotConfig
from bot.data import ensure_local_candles
from bot.rl.rewards import get_reward_function
from bot.signals import indicators


@dataclass(slots=True)
class StepInfo:
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    position: int
    price: float
    transaction_cost_bps: float
    taker_fee: float


class FuturesTradingEnv:
    action_meanings = {0: "hold", 1: "long", 2: "short", 3: "flat"}

    def __init__(
        self,
        symbol: str,
        interval: str,
        cfg: BotConfig,
        window: int | None = None,
        reward_scheme: str | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self._config = cfg
        self.window = window or cfg.rl.lookback_window
        self.reward_fn = get_reward_function(reward_scheme or cfg.rl.reward_scheme)
        self.max_steps = max_steps or cfg.rl.max_steps_per_episode
        self.base_balance = cfg.backtest.initial_balance
        # Explicitly track market impact assumptions for diagnostics/audits.
        self.transaction_cost_bps = float(cfg.backtest.slippage_bps)
        self.taker_fee = float(cfg.risk.taker_fee)
        self._load_data()
        self.observation_size = self.window + 6
        self.action_space = len(self.action_meanings)
        self._action_counts = [0 for _ in range(self.action_space)]
        self.reset()

    def _load_data(self) -> None:
        frame = ensure_local_candles(self._config, self.symbol, self.interval, min_rows=self.window + 200)
        frame = frame.sort_values("open_time").reset_index(drop=True)
        defaults = self._config.strategy.default_parameters
        fast_len = int(defaults.get("fast_ema", 13))
        slow_len = int(defaults.get("slow_ema", 34))
        rsi_len = int(defaults.get("rsi_length", 14))
        atr_len = int(defaults.get("atr_period", 14))
        frame["ema_fast"] = indicators.ema(frame["close"], fast_len)
        frame["ema_slow"] = indicators.ema(frame["close"], slow_len)
        frame["rsi"] = indicators.rsi(frame["close"], rsi_len)
        frame["atr"] = indicators.atr(frame, atr_len)
        frame = frame.dropna().reset_index(drop=True)
        if len(frame) <= self.window + 2:
            raise ValueError("Not enough data to initialize RL environment")
        self.frame = frame

    def reset(self) -> NDArray[np.float32]:
        self.index = random.randint(self.window, len(self.frame) - 2)
        self.step_count = 0
        self.position = 0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self._peak_equity = self.base_balance
        self.last_action = 0
        self._action_counts = [0 for _ in range(self.action_space)]
        return self._observation()

    def _price_window(self) -> NDArray[np.float32]:
        start = self.index - self.window
        window = self.frame.iloc[start:self.index]["close"].to_numpy(dtype=np.float32)
        baseline = window[-1]
        if baseline == 0:
            baseline = 1.0
        normalized = (window / baseline) - 1.0
        return normalized

    def _observation(self) -> NDArray[np.float32]:
        row = self.frame.iloc[self.index]
        price_features = self._price_window()
        ema_ratio = float(row["ema_fast"] / row["ema_slow"]) if row["ema_slow"] else 1.0
        rsi_norm = float(row["rsi"]) / 100.0
        atr_norm = float(row["atr"] / row["close"]) if row["close"] else 0.0
        obs = np.concatenate(
            [
                price_features,
                np.array(
                    [
                        ema_ratio,
                        rsi_norm,
                        atr_norm,
                        float(self.position),
                        self.unrealized_pnl / max(self.base_balance, 1.0),
                        self.realized_pnl / max(self.base_balance, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs.astype(np.float32)

    def step(self, action: int) -> Tuple[NDArray[np.float32], float, bool, StepInfo]:
        action = int(action) % self.action_space
        self._action_counts[action] += 1
        price = float(self.frame.iloc[self.index]["close"])
        next_price = float(self.frame.iloc[self.index + 1]["close"])
        atr = float(self.frame.iloc[self.index]["atr"])
        desired_position = self.position
        if action == 1:
            desired_position = 1
        elif action == 2:
            desired_position = -1
        elif action == 3:
            desired_position = 0
        action_change = 1.0 if desired_position != self.position else 0.0
        if desired_position != self.position and self.position != 0:
            self.realized_pnl += (price - self.entry_price) * self.position
            self.unrealized_pnl = 0.0
        if desired_position != 0 and (desired_position != self.position or self.entry_price == 0.0):
            self.entry_price = price
        self.position = desired_position
        pnl_delta = (next_price - price) * self.position
        self.unrealized_pnl += pnl_delta
        equity = self.base_balance + self.realized_pnl + self.unrealized_pnl
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (self._peak_equity - equity) / max(self._peak_equity, 1.0)
        context = {
            "pnl_delta": pnl_delta,
            "atr": atr if atr else 1.0,
            "drawdown": drawdown,
            "position": float(self.position),
            "action_change": action_change,
            # Reward shaping explicitly factors in transaction costs/fees.
            "transaction_cost_bps": self.transaction_cost_bps,
            "taker_fee": self.taker_fee,
        }
        reward = float(self.reward_fn(context))
        self.index += 1
        self.step_count += 1
        done = self.index >= len(self.frame) - 2 or self.step_count >= self.max_steps
        info = StepInfo(
            equity=equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            position=self.position,
            price=next_price,
            transaction_cost_bps=self.transaction_cost_bps,
            taker_fee=self.taker_fee,
        )
        return self._observation(), reward, done, info

    def derive_parameters(self) -> Dict[str, float]:
        total_actions = max(sum(self._action_counts), 1)
        long_bias = self._action_counts[1] / total_actions
        short_bias = self._action_counts[2] / total_actions
        defaults = self._config.strategy.default_parameters
        fast = int(defaults["fast_ema"] * (0.8 + long_bias * 0.4))
        slow = int(defaults["slow_ema"] * (1.0 + short_bias * 0.5))
        rsi_overbought = defaults["rsi_overbought"] + short_bias * 5
        rsi_oversold = defaults["rsi_oversold"] - long_bias * 5
        return {
            "fast_ema": max(5, min(60, fast)),
            "slow_ema": max(20, min(200, slow)),
            "rsi_overbought": max(50, min(80, rsi_overbought)),
            "rsi_oversold": max(20, min(50, rsi_oversold)),
            "atr_stop": max(0.5, min(3.0, defaults["atr_stop"] + short_bias * 0.5)),
            "atr_target": max(1.0, min(4.0, defaults["atr_target"] + long_bias * 0.5)),
        }


__all__ = ["FuturesTradingEnv", "StepInfo"]
