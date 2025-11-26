from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import bot.execution.live as live_mod
import bot.runner as runner_mod
from bot.core.config import load_config
from bot.learning import TradeLearningStore


def test_optimizer_overrides_blocked_without_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg = replace(cfg, runtime=replace(cfg.runtime, use_optimizer_output=False))

    def _fail(_cfg: Any, _symbol: str, _interval: str):
        raise AssertionError("load_best_params should not run when optimizer flag is off")

    monkeypatch.setattr(runner_mod, "load_best_params", _fail)

    _params, overrides, source = runner_mod.build_strategy_params_with_meta(
        cfg,
        "BTCUSDT",
        "1m",
        use_best=True,
        rl_context="backtest",
    )
    assert overrides is None
    assert source == "default"


def test_optimizer_overrides_allowed_with_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg = replace(cfg, runtime=replace(cfg.runtime, use_optimizer_output=True))
    sentinel = {"fast_ema": 8.0}

    def _best(_cfg: Any, _symbol: str, _interval: str):
        return sentinel

    monkeypatch.setattr(runner_mod, "load_best_params", _best)

    _params, overrides, source = runner_mod.build_strategy_params_with_meta(
        cfg,
        "BTCUSDT",
        "1m",
        use_best=True,
        rl_context="backtest",
    )
    assert overrides == sentinel
    assert source == "optimized"


def test_rl_policy_guardrails_reject_large_deviation(tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    allowed, reason = runner_mod.policy_guardrails(
        cfg,
        baseline_hash="hash",
        policy_baseline_hash="hash",
        param_deviation=cfg.rl.max_param_deviation_from_baseline + 0.1,
        metrics={"val_reward_mean": cfg.rl.min_validation_reward + 1.0},
    )
    assert allowed is False
    assert reason.startswith("deviation")


def test_rl_overrides_blocked_when_flag_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg = replace(cfg, runtime=replace(cfg.runtime, use_rl_policy=False))
    def _set_state(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(runner_mod.status_store, "set_rl_state", _set_state)

    def _fail(*_args: Any, **_kwargs: Any):
        raise AssertionError("_resolve_rl_store should not run when RL flag is off")

    monkeypatch.setattr(runner_mod, "_resolve_rl_store", _fail)

    result = runner_mod.load_rl_overrides(cfg, "BTCUSDT", "1m", rl_context="backtest", allow_rl=True)
    assert result is None


def test_learning_store_helper_obeys_runtime_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg_off = replace(cfg, runtime=replace(cfg.runtime, use_learning_store=False))
    created = False

    def _ctor(*_args: Any, **_kwargs: Any):
        nonlocal created
        created = True
        return object()

    monkeypatch.setattr(live_mod, "TradeLearningStore", _ctor)
    assert live_mod.resolve_learning_store(cfg_off, None) is None
    assert created is False

    cfg_on = replace(cfg, runtime=replace(cfg.runtime, use_learning_store=True))
    sentinel = object()

    def _ctor_enabled(*_args: Any, **_kwargs: Any):
        return sentinel

    monkeypatch.setattr(live_mod, "TradeLearningStore", _ctor_enabled)
    assert live_mod.resolve_learning_store(cfg_on, None) is sentinel
    provided = cast(TradeLearningStore, sentinel)
    assert live_mod.resolve_learning_store(cfg_on, provided) is provided
