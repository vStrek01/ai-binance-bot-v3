"""Actor-critic agent used by the RL trainer."""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch import optim

from bot.core import config
from bot.rl.models import PolicyNetwork, ValueNetwork


class ActorCriticAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        learning_rate: float | None = None,
        gamma: float | None = None,
        value_coef: float | None = None,
        entropy_coef: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.policy = PolicyNetwork(observation_dim, action_dim).to(self.device)
        self.value = ValueNetwork(observation_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=learning_rate or config.rl.learning_rate,
        )
        self.gamma = gamma or config.rl.gamma
        self.value_coef = value_coef or config.rl.value_coef
        self.entropy_coef = entropy_coef or config.rl.entropy_coef
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []

    def select_action(self, state_vector) -> int:
        state = torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value(state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        return int(action.item())

    def record_reward(self, reward: float) -> None:
        self.rewards.append(torch.tensor([reward], dtype=torch.float32, device=self.device))

    def finish_episode(self) -> None:
        if not self.rewards:
            return
        returns: List[torch.Tensor] = []
        R = torch.zeros(1, device=self.device)
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns_tensor = torch.cat(returns).detach().view(-1)
        values_tensor = torch.cat(self.values).view(-1)
        advantages = returns_tensor - values_tensor
        policy_losses = [-(log_prob * advantage.detach()) for log_prob, advantage in zip(self.log_probs, advantages)]
        value_loss = torch.nn.functional.mse_loss(values_tensor, returns_tensor)
        entropy_bonus = torch.cat(self.entropies).sum()
        loss = torch.stack(policy_losses).sum() + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 1.0)
        self.optimizer.step()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(), "value": self.value.state_dict()}, path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(payload["policy"])
        self.value.load_state_dict(payload["value"])


__all__ = ["ActorCriticAgent"]
