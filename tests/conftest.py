import copy
import os
from dataclasses import asdict, is_dataclass
from typing import Iterator

import pytest

from bot.core import config


def _snapshot(obj: object) -> dict:
    if is_dataclass(obj):
        return copy.deepcopy(asdict(obj))
    if hasattr(obj, "__dict__"):
        return copy.deepcopy(getattr(obj, "__dict__"))
    raise TypeError(f"Cannot snapshot config object of type {type(obj)!r}")


@pytest.fixture(autouse=True)
def restore_core_configs() -> Iterator[None]:
    risk_snapshot = _snapshot(config.risk)
    runtime_snapshot = _snapshot(config.runtime)
    rl_snapshot = _snapshot(config.rl)
    strategy_snapshot = _snapshot(config.strategy)
    env_overrides = {"BOT_CONFIRM_LIVE": os.getenv("BOT_CONFIRM_LIVE")}
    yield
    for snapshot, target in (
        (risk_snapshot, config.risk),
        (runtime_snapshot, config.runtime),
        (rl_snapshot, config.rl),
        (strategy_snapshot, config.strategy),
    ):
        for key, value in snapshot.items():
            setattr(target, key, value)
    # Restore environment overrides that tests might tweak
    for key, value in env_overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
