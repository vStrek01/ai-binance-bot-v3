from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pytest

import bot.execution.client_factory as client_factory
import bot.execution.live as live_mod
import bot.runner as runner_mod
from bot.core.config import BotConfig, load_config
from bot.learning import TradeLearningStore
from bot.rl.types import RLPolicyVersion, RLRunMetadata, compute_baseline_hash


def _with_runtime(cfg: BotConfig, **updates: Any) -> BotConfig:
    return cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update=updates)})


def _build_policy_payload(
    cfg: BotConfig,
    *,
    val_reward: float,
    deviation: float = 0.05,
) -> tuple[dict[str, float], RLPolicyVersion, RLRunMetadata]:
    baseline_hash = compute_baseline_hash(cfg.strategy.default_parameters)
    params = {
        "fast_ema": cfg.strategy.default_parameters["fast_ema"] + 1.0,
        "slow_ema": cfg.strategy.default_parameters["slow_ema"] + 2.0,
    }
    version = RLPolicyVersion(
        version_id="test-version",
        created_at=datetime.now(timezone.utc),
        run_id="run-1",
        metrics={"val_reward_mean": val_reward},
        baseline_params_hash=baseline_hash,
        param_deviation=deviation,
    )
    metadata = RLRunMetadata(
        run_id="run-1",
        policy_name="BTCUSDT:1m",
        created_at=datetime.now(timezone.utc),
        seed=42,
        env_id="test-env",
        symbol="BTCUSDT",
        interval="1m",
        episodes=5,
        baseline_params_hash=baseline_hash,
        max_param_deviation=deviation,
        train_reward_mean=val_reward,
        train_reward_std=0.1,
        val_reward_mean=val_reward,
        val_reward_std=0.1,
    )
    return params, version, metadata


class _StubRLStore:
    def __init__(self, payload: tuple[dict[str, float], RLPolicyVersion, RLRunMetadata]):
        self.payload = payload
        self.calls = 0

    def load_latest_policy_params(self, policy_name: str):  # type: ignore[override]
        self.calls += 1
        return self.payload


def test_optimizer_overrides_blocked_without_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg = _with_runtime(cfg, use_optimizer_output=False)

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
    cfg = _with_runtime(cfg, use_optimizer_output=True)
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
    cfg = _with_runtime(cfg, use_rl_policy=False)
    def _set_state(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(runner_mod.status_store, "set_rl_state", _set_state)

    def _fail(*_args: Any, **_kwargs: Any):
        raise AssertionError("_resolve_rl_store should not run when RL flag is off")

    monkeypatch.setattr(runner_mod, "_resolve_rl_store", _fail)

    result = runner_mod.load_rl_overrides(cfg, "BTCUSDT", "1m", rl_context="backtest", allow_rl=True)
    assert result is None


def test_load_rl_overrides_reports_disabled_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    captured: dict[str, Any] = {}

    def _capture(state: dict[str, Any]) -> None:
        captured.update(state)

    monkeypatch.setattr(runner_mod.status_store, "set_rl_state", _capture)
    result = runner_mod.load_rl_overrides(cfg, "BTCUSDT", "1m", rl_context="backtest", allow_rl=True)
    assert result is None
    assert captured["reason"] == "rl_disabled"
    assert captured["active"] is False


def test_load_rl_overrides_blocks_live_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg = _with_runtime(cfg, use_rl_policy=True)
    captured: dict[str, Any] = {}

    def _capture(state: dict[str, Any]) -> None:
        captured.update(state)

    def _fail_resolve(_cfg: BotConfig) -> None:
        raise AssertionError("RL store should not be resolved when context is blocked")

    monkeypatch.setattr(runner_mod.status_store, "set_rl_state", _capture)
    monkeypatch.setattr(runner_mod, "_resolve_rl_store", _fail_resolve)

    result = runner_mod.load_rl_overrides(cfg, "BTCUSDT", "1m", rl_context="live", allow_rl=False)
    assert result is None
    assert captured["reason"] == "context_blocked"
    assert captured["active"] is True


@pytest.mark.parametrize("rl_context", ["backtest", "optimizer"])
def test_rl_overrides_apply_offline_contexts(
    rl_context: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg = _with_runtime(cfg, use_rl_policy=True)
    safe_deviation = max(min(0.05, cfg.rl.max_param_deviation_from_baseline - 1e-3), 5e-4)
    payload = _build_policy_payload(
        cfg,
        val_reward=cfg.rl.min_validation_reward + 0.5,
        deviation=safe_deviation,
    )
    store = _StubRLStore(payload)
    statuses: list[dict[str, Any]] = []

    def _capture(state: dict[str, Any]) -> None:
        statuses.append(state)

    original_guardrails = runner_mod.policy_guardrails
    guardrail_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _spy_guardrails(*args: Any, **kwargs: Any):
        guardrail_calls.append((args, kwargs))
        return original_guardrails(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "_resolve_rl_store", lambda _cfg: store)
    monkeypatch.setattr(runner_mod.status_store, "set_rl_state", _capture)
    monkeypatch.setattr(runner_mod, "policy_guardrails", _spy_guardrails)

    params, overrides, source = runner_mod.build_strategy_params_with_meta(
        cfg,
        "BTCUSDT",
        "1m",
        use_best=False,
        rl_context=rl_context,
    )

    assert source == "rl_override"
    assert overrides == payload[0]
    assert params.fast_ema == int(payload[0]["fast_ema"])
    assert store.calls == 1
    assert guardrail_calls, "policy_guardrails should be invoked"
    assert statuses and statuses[-1]["applied"] is True
    assert statuses[-1]["context"] == rl_context


def test_rl_overrides_rejected_when_guardrails_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg = _with_runtime(cfg, use_rl_policy=True)
    safe_deviation = max(min(0.05, cfg.rl.max_param_deviation_from_baseline - 1e-3), 5e-4)
    payload = _build_policy_payload(
        cfg,
        val_reward=cfg.rl.min_validation_reward - 0.5,
        deviation=safe_deviation,
    )
    store = _StubRLStore(payload)
    statuses: list[dict[str, Any]] = []

    def _capture(state: dict[str, Any]) -> None:
        statuses.append(state)

    monkeypatch.setattr(runner_mod, "_resolve_rl_store", lambda _cfg: store)
    monkeypatch.setattr(runner_mod.status_store, "set_rl_state", _capture)

    params, overrides, source = runner_mod.build_strategy_params_with_meta(
        cfg,
        "BTCUSDT",
        "1m",
        use_best=False,
        rl_context="backtest",
    )

    assert overrides is None
    assert source == "default"
    assert params.fast_ema == int(cfg.strategy.default_parameters["fast_ema"])
    assert store.calls == 1
    assert statuses and statuses[-1]["reason"] == "val_reward_below_threshold"


def test_learning_store_helper_obeys_runtime_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg_off = _with_runtime(cfg, use_learning_store=False)
    created = False

    def _ctor(*_args: Any, **_kwargs: Any):
        nonlocal created
        created = True
        return object()

    monkeypatch.setattr(live_mod, "TradeLearningStore", _ctor)
    assert live_mod.resolve_learning_store(cfg_off, None) is None
    assert created is False

    cfg_on = _with_runtime(cfg, use_learning_store=True)
    sentinel = object()

    def _ctor_enabled(*_args: Any, **_kwargs: Any):
        return sentinel

    monkeypatch.setattr(live_mod, "TradeLearningStore", _ctor_enabled)
    assert live_mod.resolve_learning_store(cfg_on, None) is sentinel
    provided = cast(TradeLearningStore, sentinel)
    assert live_mod.resolve_learning_store(cfg_on, provided) is provided

def test_demo_live_profile_uses_testnet_without_confirmation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg = _with_runtime(
        cfg,
        dry_run=False,
        use_testnet=True,
        live_trading=True,
    )

    def _fail(_runtime: Any) -> None:
        raise AssertionError("Live confirmation should not run for testnet/demo-live profiles")

    monkeypatch.setattr(client_factory, "_ensure_live_confirmation", _fail)

    profile = client_factory._determine_trading_profile(cfg)
    assert profile.label == "testnet"


def test_live_profile_enforces_confirmation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = load_config(base_dir=tmp_path)
    cfg = _with_runtime(
        cfg,
        dry_run=False,
        use_testnet=False,
        live_trading=True,
    )
    called = False

    def _mark(_runtime: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(client_factory, "_ensure_live_confirmation", _mark)
    profile = client_factory._determine_trading_profile(cfg)
    assert profile.label == "live"
    assert called is True
