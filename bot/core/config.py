"""Thin adapter exposing the canonical config schema throughout the bot."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from infra.config_loader import load_config as _load_app_config
from infra.config_schema import (
    AppConfig,
    ExternalSignalConfig,
    MultiTimeframeConfig,
    OptimizerConfig,
    PathsConfig,
    ReinforcementConfig,
    RLConfig,
    RiskConfig,
    RuntimeConfig,
    SizingConfig,
    StrategyConfig,
    UniverseConfig,
)

BotConfig = AppConfig

__all__ = [
    "BotConfig",
    "ExternalSignalConfig",
    "MultiTimeframeConfig",
    "OptimizerConfig",
    "PathsConfig",
    "ReinforcementConfig",
    "RLConfig",
    "RiskConfig",
    "RuntimeConfig",
    "SizingConfig",
    "StrategyConfig",
    "UniverseConfig",
    "ensure_directories",
    "load_config",
]


def ensure_directories(paths: PathsConfig, extra: Iterable[Path] | None = None) -> None:
    """Create the common directory structure if it does not already exist."""
    targets = [paths.data_dir, paths.results_dir, paths.optimization_dir, paths.log_dir]
    if extra:
        targets.extend(extra)
    for folder in targets:
        folder.mkdir(parents=True, exist_ok=True)


def load_config(*, base_dir: Path | None = None) -> BotConfig:
    """Load the canonical AppConfig and expose it under the legacy name."""

    return _load_app_config(base_dir=base_dir)
