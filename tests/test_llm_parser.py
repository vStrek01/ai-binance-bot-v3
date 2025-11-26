from ai.decision_schema import LlmDecision, parse_llm_response


def test_parse_valid_long():
    raw = '{"action": "LONG", "confidence": 80, "reason": "trend"}'
    decision = parse_llm_response(raw)
    assert isinstance(decision, LlmDecision)
    assert decision.action == "LONG"
    assert decision.confidence == 80
    assert decision.reason == "trend"


def test_parse_accepts_extra_fields():
    raw = '{"action": "FLAT", "confidence": 55, "reason": "chop", "foo": "bar"}'
    decision = parse_llm_response(raw)
    assert decision is not None
    assert decision.action == "FLAT"


def test_parse_missing_field_returns_none():
    raw = '{"action": "LONG", "reason": "trend"}'
    assert parse_llm_response(raw) is None


def test_parse_wrong_type_returns_none():
    raw = '{"action": 123, "confidence": "high", "reason": "bad"}'
    assert parse_llm_response(raw) is None


def test_parse_invalid_json_returns_none():
    raw = '{"action": "LONG",'
    assert parse_llm_response(raw) is None


def test_parse_non_json_returns_none():
    raw = "just words"
    assert parse_llm_response(raw) is None


def test_parse_none_returns_none():
    assert parse_llm_response(None) is None
