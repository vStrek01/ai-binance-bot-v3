from __future__ import annotations

from bot import config
from bot.runner import _build_strategy_params, _rl_guardrails_allow


def _baseline_params() -> dict[str, float]:
    return {key: float(value) for key, value in config.strategy.default_parameters.items()}


def test_rl_guardrails_allow_values_within_tolerance() -> None:
    config.rl.max_param_deviation_from_baseline = 0.1
    params = {key: value * 1.05 for key, value in _baseline_params().items()}
    allowed, reason = _rl_guardrails_allow(params)
    assert allowed is True
    assert reason == ""


def test_rl_guardrails_reject_values_beyond_tolerance() -> None:
    config.rl.max_param_deviation_from_baseline = 0.1
    params = _baseline_params()
    params["fast_ema"] = params["fast_ema"] * 2.0
    allowed, reason = _rl_guardrails_allow(params)
    assert allowed is False
    assert "fast_ema" in reason
    assert "deviation" in reason


def test_build_strategy_params_skips_rl_when_live_disallowed(monkeypatch) -> None:
    config.runtime.use_rl_policy = True
    config.rl.enabled = True
    config.rl.apply_to_live = False

    def forbidden_store() -> None:  # pragma: no cover - should never be called
        raise AssertionError("RL store should not be resolved when apply_to_live=False")

    monkeypatch.setattr("bot.runner._resolve_rl_store", forbidden_store)
    params = _build_strategy_params("BTCUSDT", "1m", use_best=False, rl_context="live")
    assert params is not None
