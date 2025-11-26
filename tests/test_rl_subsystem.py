from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

import bot.runner as runner
from bot.core.config import BotConfig, load_config
from bot.rl.policy_store import RLPolicyStore, _policy_name
from bot.rl.trainer import RLTrainer
from bot.rl.types import RLPolicyVersion, RLRunMetadata, compute_baseline_hash
from bot.status import status_store


def _test_config(tmp_path: Path, *, min_val_reward: float | None = None) -> BotConfig:
    cfg = load_config(base_dir=tmp_path)
    if min_val_reward is not None:
        cfg = replace(cfg, rl=replace(cfg.rl, min_validation_reward=min_val_reward))
    cfg = replace(cfg, runtime=replace(cfg.runtime, use_rl_policy=True))
    status_store.configure(log_dir=tmp_path / "logs")
    return cfg


def _sample_metadata(policy_name: str, baseline_hash: str, cfg: BotConfig, run_id: str) -> RLRunMetadata:
    return RLRunMetadata(
        run_id=run_id,
        policy_name=policy_name,
        created_at=datetime.now(timezone.utc),
        seed=0,
        env_id="TestEnv",
        symbol="BTCUSDT",
        interval="1h",
        episodes=1,
        baseline_params_hash=baseline_hash,
        max_param_deviation=cfg.rl.max_param_deviation_from_baseline,
        train_reward_mean=1.0,
        train_reward_std=0.0,
        val_reward_mean=1.0,
        val_reward_std=0.0,
    )


def test_policy_store_versioning(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path)
    store = RLPolicyStore(cfg, root=tmp_path / "policies")
    policy_name = _policy_name("BTCUSDT", "1h")
    baseline_hash = compute_baseline_hash(cfg.strategy.default_parameters)
    model = torch.nn.Linear(1, 1)

    meta1 = _sample_metadata(policy_name, baseline_hash, cfg, run_id="run1")
    version1 = store.save_policy(
        policy_name,
        model,
        meta1,
        param_deviation=0.1,
        metrics={"val_reward_mean": 1.0},
        config_snapshot={"params": {"fast_ema": 10}},
    )

    meta2 = _sample_metadata(policy_name, baseline_hash, cfg, run_id="run2")
    version2 = store.save_policy(
        policy_name,
        model,
        meta2,
        param_deviation=0.05,
        metrics={"val_reward_mean": 2.0},
        config_snapshot={"params": {"fast_ema": 20}},
    )

    versions = store.list_policies(policy_name)
    assert [v.version_id for v in versions] == [version1.version_id, version2.version_id]

    params, latest_version, run_metadata = store.load_latest_policy_params(policy_name)  # type: ignore[misc]
    assert latest_version.version_id == version2.version_id
    assert params["fast_ema"] == 20
    metadata_path = store.root / policy_name / version2.version_id / "metadata.json"
    payload = json.loads(metadata_path.read_text())
    assert payload["policy_version"]["baseline_params_hash"] == baseline_hash
    assert payload["run_metadata"]["run_id"] == "run2"
    assert run_metadata.run_id == "run2"


def test_rl_policy_guardrails(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path, min_val_reward=1.5)
    baseline_hash = compute_baseline_hash(cfg.strategy.default_parameters)
    policy_name = _policy_name("BTCUSDT", "1h")
    version = RLPolicyVersion(
        version_id="v1",
        created_at=datetime.now(timezone.utc),
        run_id="runA",
        metrics={"val_reward_mean": 1.0},
        baseline_params_hash=baseline_hash,
        param_deviation=0.1,
    )
    run_metadata = _sample_metadata(policy_name, baseline_hash, cfg, run_id="runA")
    payload = ({"fast_ema": cfg.strategy.default_parameters["fast_ema"]}, version, run_metadata)

    class FakeStore:
        def __init__(self, item):
            self.item = item

        def load_latest_policy_params(self, policy_name: str):
            return self.item

    original = runner._resolve_rl_store
    runner._resolve_rl_store = lambda _cfg: FakeStore(payload)
    try:
        rejected = runner._load_rl_overrides(cfg, "BTCUSDT", "1h", rl_context="test", allow_rl=True)
        assert rejected is None
    finally:
        runner._resolve_rl_store = original

    good_version = replace(version, metrics={"val_reward_mean": 2.0})
    good_payload = ({"fast_ema": cfg.strategy.default_parameters["fast_ema"]}, good_version, run_metadata)
    runner._resolve_rl_store = lambda _cfg: FakeStore(good_payload)
    try:
        accepted = runner._load_rl_overrides(cfg, "BTCUSDT", "1h", rl_context="test", allow_rl=True)
        assert accepted is not None
    finally:
        runner._resolve_rl_store = original


class _FakeEnv:
    def __init__(self) -> None:
        self.symbol = "BTCUSDT"
        self.interval = "1h"
        self._step = 0

    def reset(self):
        self._step = 0
        return [0.0]

    def step(self, action):
        self._step += 1
        done = self._step >= 2
        return [0.0], 1.0, done, {}

    def derive_parameters(self):
        return {"fast_ema": 13, "slow_ema": 34}


class _FakeAgent:
    def __init__(self) -> None:
        self.policy = torch.nn.Linear(1, 1)

    def select_action(self, state_vector):
        return 0

    def select_action_eval(self, state_vector):
        return 0

    def record_reward(self, reward: float) -> None:
        pass

    def finish_episode(self) -> None:
        pass

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)


def test_trainer_writes_diagnostics(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path)
    env = _FakeEnv()
    agent = _FakeAgent()
    trainer = RLTrainer(env, agent, cfg, checkpoint_dir=tmp_path / "ckpt", results_dir=tmp_path / "runs")
    result = trainer.train(episodes=2, validation_episodes=1)

    run_dirs = list((tmp_path / "runs").iterdir())
    assert run_dirs, "expected trainer to write a run directory"
    metadata_path = run_dirs[0] / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["train_reward_mean"] == 2.0
    assert metadata["val_reward_mean"] == 2.0

    policy_root = cfg.paths.optimization_dir / cfg.rl.policy_dir_name
    versions = list(policy_root.rglob("metadata.json"))
    assert versions, "expected a saved policy version"

