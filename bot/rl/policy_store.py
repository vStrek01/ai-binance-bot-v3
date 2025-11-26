"""Storage helpers for RL-derived strategy parameters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from bot.core.config import BotConfig, ensure_directories
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger

logger = get_logger(__name__)


class RLPolicyStore:
    def __init__(self, cfg: BotConfig, path: Optional[Path] = None) -> None:
        ensure_directories(cfg.paths, extra=[cfg.paths.optimization_dir])
        self.path = path or (cfg.paths.optimization_dir / "rl_policies.json")
        self._payload: Dict[str, Dict[str, float]] = self._load()

    def _load(self) -> Dict[str, Dict[str, float]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            logger.warning("RL policy store corrupted; resetting %s", self.path)
            return {}
        return data if isinstance(data, dict) else {}

    def _key(self, symbol: str, interval: str) -> str:
        return f"{symbol.upper()}:{interval}"

    def get(self, symbol: str, interval: str) -> Optional[Dict[str, float]]:
        return self._payload.get(self._key(symbol, interval))

    def save(self, symbol: str, interval: str, params: Dict[str, float]) -> None:
        key = self._key(symbol, interval)
        self._payload[key] = params
        atomic_write_text(self.path, json.dumps(self._payload, indent=2))
        logger.info("Saved RL-derived parameters for %s %s", symbol, interval)


__all__ = ["RLPolicyStore"]
