"""Training orchestration for the futures RL environment."""
from __future__ import annotations

import json
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from bot.core.config import BotConfig, ensure_directories
from bot.rl.agents import ActorCriticAgent
from bot.rl.env import FuturesTradingEnv
from bot.rl.policy_store import RLPolicyStore, _policy_name
from bot.rl.types import RLRunMetadata, compute_baseline_hash, compute_param_deviation
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger, get_run_context

logger = get_logger(__name__)


def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


class RLTrainer:
    def __init__(
        self,
        env: FuturesTradingEnv,
        agent: ActorCriticAgent,
        cfg: BotConfig,
        checkpoint_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        ensure_directories(cfg.paths, extra=[cfg.paths.optimization_dir, cfg.paths.results_dir])
        self.checkpoint_dir = checkpoint_dir or (cfg.paths.optimization_dir / cfg.rl.checkpoint_dir_name)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = results_dir or (cfg.paths.results_dir / cfg.rl.results_dir_name)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._config = cfg

    def _run_episode(self) -> float:
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.record_reward(reward)
            state = next_state
            episode_reward += reward
        self.agent.finish_episode()
        return episode_reward

    def _evaluate_episode(self) -> float:
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = self.agent.select_action_eval(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            episode_reward += reward
        return episode_reward

    def _serialize_rewards(self, path: Path, values: List[float]) -> None:
        atomic_write_text(path, json.dumps(values, indent=2))

    def _create_run_metadata(
        self,
        *,
        run_id: str,
        episodes: int,
        train_rewards: List[float],
        val_rewards: List[float],
        baseline_hash: str,
    ) -> RLRunMetadata:
        return RLRunMetadata(
            run_id=run_id,
            policy_name=_policy_name(self.env.symbol, self.env.interval),
            created_at=datetime.now(timezone.utc),
            seed=0,
            env_id=self.env.__class__.__name__,
            symbol=self.env.symbol,
            interval=self.env.interval,
            episodes=int(episodes),
            baseline_params_hash=baseline_hash,
            max_param_deviation=self._config.rl.max_param_deviation_from_baseline,
            train_reward_mean=_mean(train_rewards),
            train_reward_std=_std(train_rewards),
            val_reward_mean=_mean(val_rewards),
            val_reward_std=_std(val_rewards),
        )

    def _persist_run(
        self,
        *,
        run_id: str,
        train_rewards: List[float],
        val_rewards: List[float],
        metadata: RLRunMetadata,
    ) -> None:
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(run_dir / "metadata.json", json.dumps(metadata.to_json(), indent=2))
        self._serialize_rewards(run_dir / "train_rewards.json", train_rewards)
        if val_rewards:
            self._serialize_rewards(run_dir / "val_rewards.json", val_rewards)

    def train(
        self,
        episodes: int,
        checkpoint_interval: Optional[int] = None,
        validation_episodes: Optional[int] = None,
    ) -> Dict[str, float]:
        rewards: List[float] = []
        best_reward = float("-inf")
        checkpoint_every = checkpoint_interval or self._config.rl.checkpoint_interval
        validation_runs = validation_episodes if validation_episodes is not None else self._config.rl.validation_episodes
        context = get_run_context()
        run_id = context.run_id if context else uuid.uuid4().hex
        for episode in range(episodes):
            episode_reward = self._run_episode()
            rewards.append(episode_reward)
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save(self.checkpoint_dir / "best.pt")
            if (episode + 1) % checkpoint_every == 0:
                self.agent.save(self.checkpoint_dir / f"episode_{episode + 1}.pt")
                logger.info(
                    "rl_episode_checkpoint",
                    extra={
                        "episode": episode + 1,
                        "episodes": episodes,
                        "reward": episode_reward,
                        "avg_reward": _mean(rewards),
                        "run_id": run_id,
                    },
                )
        val_rewards: List[float] = []
        if validation_runs > 0:
            for _ in range(validation_runs):
                val_rewards.append(self._evaluate_episode())
        baseline = dict(self._config.strategy.default_parameters)
        derived_params = self.env.derive_parameters()
        baseline_hash = compute_baseline_hash(baseline)
        param_deviation = compute_param_deviation(baseline, derived_params)
        run_metadata = self._create_run_metadata(
            run_id=run_id,
            episodes=episodes,
            train_rewards=rewards,
            val_rewards=val_rewards,
            baseline_hash=baseline_hash,
        )
        self._persist_run(run_id=run_id, train_rewards=rewards, val_rewards=val_rewards, metadata=run_metadata)
        RLPolicyStore(self._config).save_policy(
            run_metadata.policy_name,
            self.agent.policy,
            run_metadata,
            param_deviation,
            metrics={
                "train_reward_mean": run_metadata.train_reward_mean,
                "train_reward_std": run_metadata.train_reward_std,
                "val_reward_mean": run_metadata.val_reward_mean,
                "val_reward_std": run_metadata.val_reward_std,
            },
            config_snapshot={
                "params": derived_params,
                "baseline": baseline,
            },
        )
        logger.info(
            "rl_training_complete",
            extra={
                "run_id": run_id,
                "episodes": episodes,
                "train_reward_mean": run_metadata.train_reward_mean,
                "val_reward_mean": run_metadata.val_reward_mean,
                "param_deviation": param_deviation,
            },
        )
        return {
            "episodes": float(episodes),
            "avg_reward": _mean(rewards),
            "best_reward": best_reward,
            "val_reward": _mean(val_rewards),
        }


__all__ = ["RLTrainer"]
