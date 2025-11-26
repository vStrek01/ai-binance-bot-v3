import json

from core.llm_adapter import LLMAdapter
from core.models import Side


def test_valid_llm_output():
    adapter = LLMAdapter(llm_client=lambda prompt: json.dumps({"action": "LONG", "confidence": 0.8, "reason": "trend"}))
    signal = adapter.infer({"x": 1})
    assert signal.action == Side.LONG
    assert signal.confidence == 0.8


def test_invalid_llm_output_returns_flat():
    adapter = LLMAdapter(llm_client=lambda prompt: "not json")
    signal = adapter.infer({"x": 1})
    assert signal.action == Side.FLAT
    assert signal.confidence == 0.0
