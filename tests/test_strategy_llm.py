from __future__ import annotations

from core.models import LLMSignal, Signal, Side
from core.strategy import IndicatorConfig, Strategy


def _strategy() -> Strategy:
    return Strategy(indicator_config=IndicatorConfig())


def test_llm_cannot_override_flat_baseline() -> None:
    strat = _strategy()
    indicator = Signal(action=Side.FLAT, confidence=0.0, reason="flat")
    llm = LLMSignal(action=Side.LONG, confidence=0.9, reason="llm")

    fused = strat._fuse_signals(indicator, llm)

    assert fused.action is Side.FLAT
    assert fused.reason == "flat"


def test_llm_conflict_flattens_signal() -> None:
    strat = _strategy()
    indicator = Signal(action=Side.LONG, confidence=0.7, reason="trend")
    llm = LLMSignal(action=Side.SHORT, confidence=0.8, reason="llm")

    fused = strat._fuse_signals(indicator, llm)

    assert fused.action is Side.FLAT
    assert fused.reason == "llm_conflict"


def test_llm_agreement_boosts_confidence_with_cap() -> None:
    strat = _strategy()
    indicator = Signal(action=Side.LONG, confidence=0.8, reason="trend")
    llm = LLMSignal(action=Side.LONG, confidence=1.0, reason="llm")

    fused = strat._fuse_signals(indicator, llm)

    assert fused.action is Side.LONG
    assert 0.8 <= fused.confidence <= 1.0
    assert fused.reason == "agreement"
