from __future__ import annotations

import json
from typing import Literal, Optional

from infra.pydantic_guard import ensure_pydantic_v2
from pydantic import BaseModel, ValidationError, conint

ensure_pydantic_v2()


class LlmDecision(BaseModel):
    action: Literal["LONG", "SHORT", "FLAT"]
    confidence: conint(ge=0, le=100)
    reason: str


def parse_llm_response(raw: str) -> Optional[LlmDecision]:
    """Parse and validate raw LLM output. Returns None on any issue."""

    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    try:
        return LlmDecision(**payload)
    except ValidationError:
        return None
