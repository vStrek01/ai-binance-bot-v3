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


def _maybe_snapshot(name: str) -> tuple[str, dict] | None:
    target = getattr(config, name, None)
    if target is None:
        return None
    return name, _snapshot(target)


@pytest.fixture(autouse=True)
def restore_core_configs() -> Iterator[None]:
    tracked = filter(None, (_maybe_snapshot(name) for name in ("risk", "runtime", "rl", "strategy")))
    snapshots = {name: snapshot for name, snapshot in tracked}
    env_overrides = {"BOT_CONFIRM_LIVE": os.getenv("BOT_CONFIRM_LIVE")}
    yield
    for name, payload in snapshots.items():
        target = getattr(config, name, None)
        if target is None:
            continue
        for key, value in payload.items():
            setattr(target, key, value)
    # Restore environment overrides that tests might tweak
    for key, value in env_overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
