"""Versioned RL policy storage with metadata guardrails."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from bot.core.config import BotConfig, ensure_directories
from bot.rl.types import RLPolicyVersion, RLRunMetadata
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger

logger = get_logger(__name__)


def _policy_name(symbol: str, interval: str) -> str:
    return f"{symbol.upper()}:{interval}"


def _serialize(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


class RLPolicyStore:
    def __init__(self, cfg: BotConfig, root: Optional[Path] = None) -> None:
        ensure_directories(cfg.paths, extra=[cfg.paths.optimization_dir])
        self.root = root or (cfg.paths.optimization_dir / cfg.rl.policy_dir_name)
        self.root.mkdir(parents=True, exist_ok=True)

    def _policy_dir(self, policy_name: str) -> Path:
        return self.root / policy_name

    def _version_dir(self, policy_name: str, version_id: str) -> Path:
        return self._policy_dir(policy_name) / version_id

    def save_policy(
        self,
        policy_name: str,
        model: torch.nn.Module,
        run_metadata: RLRunMetadata,
        param_deviation: float,
        metrics: Dict[str, float],
        *,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> RLPolicyVersion:
        version_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc)
        version = RLPolicyVersion(
            version_id=version_id,
            created_at=created_at,
            run_id=run_metadata.run_id,
            metrics=metrics,
            baseline_params_hash=run_metadata.baseline_params_hash,
            param_deviation=param_deviation,
        )
        version_dir = self._version_dir(policy_name, version_id)
        version_dir.mkdir(parents=True, exist_ok=True)
        policy_path = version_dir / "policy.pt"
        torch.save(model.state_dict(), policy_path)
        metadata_path = version_dir / "metadata.json"
        metadata_payload = {
            "policy_version": version.to_json(),
            "run_metadata": run_metadata.to_json(),
        }
        atomic_write_text(metadata_path, _serialize(metadata_payload))
        snapshot_path = version_dir / "config_snapshot.json"
        atomic_write_text(snapshot_path, _serialize(config_snapshot or {}))
        logger.info(
            "Saved RL policy %s version=%s deviation=%.3f val_reward=%.4f",
            policy_name,
            version_id,
            param_deviation,
            metrics.get("val_reward_mean", float("nan")),
        )
        return version

    def list_policies(self, policy_name: str) -> List[RLPolicyVersion]:
        policy_dir = self._policy_dir(policy_name)
        if not policy_dir.exists():
            return []
        versions: List[RLPolicyVersion] = []
        for candidate in policy_dir.iterdir():
            if not candidate.is_dir():
                continue
            metadata_path = candidate / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                payload = json.loads(metadata_path.read_text())
                version = RLPolicyVersion.from_json(payload["policy_version"])
            except Exception:  # noqa: BLE001 - corrupt version skipped
                logger.warning("Skipping invalid RL policy metadata at %s", metadata_path)
                continue
            versions.append(version)
        versions.sort(key=lambda item: item.created_at)
        return versions

    def _load_payload(
        self, policy_name: str, version: RLPolicyVersion
    ) -> Optional[Tuple[Path, Dict[str, Any], RLRunMetadata]]:
        version_dir = self._version_dir(policy_name, version.version_id)
        policy_path = version_dir / "policy.pt"
        metadata_path = version_dir / "metadata.json"
        snapshot_path = version_dir / "config_snapshot.json"
        if not (policy_path.exists() and metadata_path.exists() and snapshot_path.exists()):
            return None
        try:
            metadata_payload = json.loads(metadata_path.read_text())
            run_metadata = RLRunMetadata.from_json(metadata_payload["run_metadata"])
            config_snapshot = json.loads(snapshot_path.read_text())
        except Exception:  # noqa: BLE001
            logger.warning("Failed to parse RL policy payload for %s %s", policy_name, version.version_id)
            return None
        return policy_path, config_snapshot, run_metadata

    def load_latest_policy(
        self, policy_name: str, model_factory
    ) -> Optional[Tuple[torch.nn.Module, RLPolicyVersion, Dict[str, Any], RLRunMetadata]]:
        versions = self.list_policies(policy_name)
        if not versions:
            return None
        latest = versions[-1]
        payload = self._load_payload(policy_name, latest)
        if payload is None:
            return None
        policy_path, config_snapshot, run_metadata = payload
        model = model_factory()
        model.load_state_dict(torch.load(policy_path, map_location="cpu"))
        return model, latest, config_snapshot, run_metadata

    def load_latest_policy_params(
        self, policy_name: str
    ) -> Optional[Tuple[Dict[str, Any], RLPolicyVersion, RLRunMetadata]]:
        versions = self.list_policies(policy_name)
        if not versions:
            return None
        latest = versions[-1]
        payload = self._load_payload(policy_name, latest)
        if payload is None:
            return None
        _, config_snapshot, run_metadata = payload
        params = config_snapshot.get("params") if isinstance(config_snapshot, dict) else None
        if not isinstance(params, dict):
            return None
        return params, latest, run_metadata


__all__ = ["RLPolicyStore", "_policy_name"]
