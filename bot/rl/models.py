"""Neural network building blocks for RL agents."""
# EXPERIMENTAL — DO NOT ENABLE FOR LIVE TRADING WITHOUT SEPARATE VALIDATION.
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - torch required by design
    raise RuntimeError("PyTorch is required for bot.rl modules. Install torch>=2.1 to continue.") from exc


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


__all__ = ["PolicyNetwork", "ValueNetwork", "torch"]
