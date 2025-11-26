from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from bot.core.config import BotConfig, load_config
from bot.rl.types import compute_baseline_hash
from bot.runner import _policy_guardrails, _rl_guardrails_allow


def _test_config(
    tmp_path: Path,
    *,
    deviation: float = 0.1,
    min_val_reward: float = 0.5,
) -> BotConfig:
    cfg = load_config(base_dir=tmp_path)
    cfg = replace(
        cfg,
        rl=replace(
            cfg.rl,
            max_param_deviation_from_baseline=deviation,
            min_validation_reward=min_val_reward,
        ),
    )
    return cfg


def _baseline_params(cfg: BotConfig) -> dict[str, float]:
    return {key: float(value) for key, value in cfg.strategy.default_parameters.items()}


def test_rl_guardrails_allow_values_within_tolerance(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path, deviation=0.1)
    params = {key: value * 1.05 for key, value in _baseline_params(cfg).items()}
    allowed, reason = _rl_guardrails_allow(params, cfg)
    assert allowed is True
    assert reason == ""


def test_rl_guardrails_reject_values_beyond_tolerance(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path, deviation=0.1)
    params = _baseline_params(cfg)
    params["fast_ema"] = params["fast_ema"] * 2.0
    allowed, reason = _rl_guardrails_allow(params, cfg)
    assert allowed is False
    assert "deviation" in reason


def test_policy_guardrails_accepts_valid_policy(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path, deviation=0.2, min_val_reward=0.5)
    baseline_hash = compute_baseline_hash(cfg.strategy.default_parameters)
    allowed, reason = _policy_guardrails(
        cfg,
        baseline_hash=baseline_hash,
        policy_baseline_hash=baseline_hash,
        param_deviation=0.1,
        metrics={"val_reward_mean": 0.6},
    )
    assert allowed is True
    assert reason == ""


def test_policy_guardrails_rejects_baseline_mismatch(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path)
    baseline_hash = compute_baseline_hash(cfg.strategy.default_parameters)
    allowed, reason = _policy_guardrails(
        cfg,
        baseline_hash=baseline_hash,
        policy_baseline_hash="different",
        param_deviation=0.05,
        metrics={"val_reward_mean": cfg.rl.min_validation_reward + 0.1},
    )
    assert allowed is False
    assert reason == "baseline_mismatch"


def test_policy_guardrails_rejects_high_deviation(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path, deviation=0.1)
    baseline_hash = compute_baseline_hash(cfg.strategy.default_parameters)
    allowed, reason = _policy_guardrails(
        cfg,
        baseline_hash=baseline_hash,
        policy_baseline_hash=baseline_hash,
        param_deviation=cfg.rl.max_param_deviation_from_baseline + 0.05,
        metrics={"val_reward_mean": cfg.rl.min_validation_reward + 0.1},
    )
    assert allowed is False
    assert reason.startswith("deviation_")


def test_policy_guardrails_rejects_low_validation_reward(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path, min_val_reward=1.0)
    baseline_hash = compute_baseline_hash(cfg.strategy.default_parameters)
    allowed, reason = _policy_guardrails(
        cfg,
        baseline_hash=baseline_hash,
        policy_baseline_hash=baseline_hash,
        param_deviation=0.01,
        metrics={"val_reward_mean": 0.5},
    )
    assert allowed is False
    assert reason == "val_reward_below_threshold"


def test_policy_guardrails_require_validation_metric(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path)
    baseline_hash = compute_baseline_hash(cfg.strategy.default_parameters)
    allowed, reason = _policy_guardrails(
        cfg,
        baseline_hash=baseline_hash,
        policy_baseline_hash=baseline_hash,
        param_deviation=0.01,
        metrics={},
    )
    assert allowed is False
    assert reason == "missing_validation_metrics"
