"""Training orchestration for the futures RL environment."""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Dict, List

from bot.core import config
from bot.rl.agents import ActorCriticAgent
from bot.rl.env import FuturesTradingEnv
from bot.rl.policy_store import RLPolicyStore
from bot.utils.logger import get_logger

logger = get_logger(__name__)


class RLTrainer:
    def __init__(
        self,
        env: FuturesTradingEnv,
        agent: ActorCriticAgent,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.checkpoint_dir = checkpoint_dir or config.rl.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, episodes: int, checkpoint_interval: int = 25) -> Dict[str, float]:
        rewards: List[float] = []
        best_reward = float("-inf")
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.record_reward(reward)
                state = next_state
                episode_reward += reward
                steps += 1
            self.agent.finish_episode()
            rewards.append(episode_reward)
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save(self.checkpoint_dir / "best.pt")
            if (episode + 1) % checkpoint_interval == 0:
                self.agent.save(self.checkpoint_dir / f"episode_{episode + 1}.pt")
                logger.info(
                    "RL episode %s/%s reward=%.4f avg=%.4f",
                    episode + 1,
                    episodes,
                    episode_reward,
                    statistics.mean(rewards) if rewards else 0.0,
                )
        params = self.env.derive_parameters()
        RLPolicyStore().save(self.env.symbol, self.env.interval, params)
        return {
            "episodes": float(episodes),
            "avg_reward": statistics.mean(rewards) if rewards else 0.0,
            "best_reward": best_reward,
        }


__all__ = ["RLTrainer"]
