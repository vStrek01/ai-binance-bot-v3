"""Typed metadata structures for RL training runs and policy versions."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict


def _serialize_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(payload: str) -> datetime:
    return datetime.fromisoformat(payload)


def compute_baseline_hash(params: Dict[str, Any]) -> str:
    """Compute a deterministic hash for the baseline parameter dictionary."""

    serialized = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_param_deviation(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """Return the maximum relative deviation between baseline and candidate params."""

    max_dev = 0.0
    for key, base_value in baseline.items():
        if key not in candidate:
            continue
        try:
            base = float(base_value)
            cand = float(candidate[key])
        except (TypeError, ValueError):
            continue
        denominator = abs(base) if abs(base) > 1e-9 else 1.0
        deviation = abs(cand - base) / denominator
        max_dev = max(max_dev, deviation)
    return max_dev


@dataclass(frozen=True)
class RLRunMetadata:
    run_id: str
    policy_name: str
    created_at: datetime
    seed: int
    env_id: str
    symbol: str
    interval: str
    episodes: int
    baseline_params_hash: str
    max_param_deviation: float
    train_reward_mean: float
    train_reward_std: float
    val_reward_mean: float
    val_reward_std: float

    def to_json(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["created_at"] = _serialize_datetime(self.created_at)
        return payload

    @staticmethod
    def from_json(payload: Dict[str, Any]) -> "RLRunMetadata":
        return RLRunMetadata(
            run_id=str(payload["run_id"]),
            policy_name=str(payload["policy_name"]),
            created_at=_parse_datetime(str(payload["created_at"])),
            seed=int(payload["seed"]),
            env_id=str(payload["env_id"]),
            symbol=str(payload["symbol"]),
            interval=str(payload["interval"]),
            episodes=int(payload["episodes"]),
            baseline_params_hash=str(payload["baseline_params_hash"]),
            max_param_deviation=float(payload["max_param_deviation"]),
            train_reward_mean=float(payload["train_reward_mean"]),
            train_reward_std=float(payload["train_reward_std"]),
            val_reward_mean=float(payload.get("val_reward_mean", 0.0)),
            val_reward_std=float(payload.get("val_reward_std", 0.0)),
        )


@dataclass(frozen=True)
class RLPolicyVersion:
    version_id: str
    created_at: datetime
    run_id: str
    metrics: Dict[str, float]
    baseline_params_hash: str
    param_deviation: float

    def to_json(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["created_at"] = _serialize_datetime(self.created_at)
        return payload

    @staticmethod
    def from_json(payload: Dict[str, Any]) -> "RLPolicyVersion":
        return RLPolicyVersion(
            version_id=str(payload["version_id"]),
            created_at=_parse_datetime(str(payload["created_at"])),
            run_id=str(payload["run_id"]),
            metrics={str(k): float(v) for k, v in (payload.get("metrics") or {}).items()},
            baseline_params_hash=str(payload["baseline_params_hash"]),
            param_deviation=float(payload["param_deviation"]),
        )


__all__ = [
    "compute_baseline_hash",
    "compute_param_deviation",
    "RLPolicyVersion",
    "RLRunMetadata",
]
