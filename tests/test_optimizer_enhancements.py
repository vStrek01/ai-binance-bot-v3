from dataclasses import replace

import pytest

from bot.core.config import load_config
from bot.optimizer import Optimizer


@pytest.fixture
def optimizer_config(tmp_path):
    base_cfg = load_config(base_dir=tmp_path)
    return base_cfg


class _DummyClient:
    def exchange_info(self):  # pragma: no cover - simple stub
        return {"symbols": []}

    def leverage_bracket(self):  # pragma: no cover - simple stub
        return []


def _patch_exchange(monkeypatch):
    monkeypatch.setattr("bot.optimizer.build_data_client", lambda cfg: _DummyClient())
    monkeypatch.setattr("bot.optimizer.ExchangeInfoManager.refresh", lambda self, force=False: None)


def test_random_param_sets_are_deterministic(optimizer_config, monkeypatch):
    _patch_exchange(monkeypatch)
    strategy = replace(
        optimizer_config.strategy,
        parameter_space={
            "fast_ema": [8, 13],
            "slow_ema": [34, 55],
            "atr_target": [1.8, 2.0],
        },
    )
    optimizer_cfg = replace(
        optimizer_config.optimizer,
        search_mode="random",
        random_subset=2,
        random_seed=7,
        enable_parallel=False,
        randomize=False,
    )
    cfg = replace(optimizer_config, strategy=strategy, optimizer=optimizer_cfg)
    opt = Optimizer(["BTCUSDT"], ["1m"], cfg=cfg)

    first = opt._param_sets()
    second = opt._param_sets()

    assert first == second
    assert len(first) == 2


def test_early_stop_limits_evaluations(optimizer_config, monkeypatch):
    _patch_exchange(monkeypatch)
    strategy = replace(
        optimizer_config.strategy,
        parameter_space={
            "fast_ema": [8, 13],
            "slow_ema": [34, 55],
        },
    )
    optimizer_cfg = replace(
        optimizer_config.optimizer,
        enable_parallel=False,
        search_mode="grid",
        randomize=False,
        early_stop_patience=2,
        min_improvement=5.0,
    )
    cfg = replace(optimizer_config, strategy=strategy, optimizer=optimizer_cfg)

    opt = Optimizer(["BTCUSDT"], ["1m"], cfg=cfg)

    call_values = [10.0, 11.0, 11.2, 50.0]
    calls = []

    def fake_run_task(task):
        value = call_values.pop(0)
        calls.append(value)
        return {
            "symbol": task[0],
            "timeframe": task[1],
            "params": task[2],
            "metrics": {"total_pnl": value},
        }

    monkeypatch.setattr("bot.optimizer._run_task", fake_run_task)

    results = opt.run()

    assert len(results) == len(calls)
    assert calls == [10.0, 11.0, 11.2]
    assert pytest.approx(results[0]["metrics"]["total_pnl"]) == 11.2
    assert call_values == [50.0]