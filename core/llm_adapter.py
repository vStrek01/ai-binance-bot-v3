from __future__ import annotations

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError, field_validator

from core.models import LLMSignal, Side
from infra.logging import logger


class LLMResponse(BaseModel):
    action: Side
    confidence: float
    reason: str

    @field_validator("confidence")
    @classmethod
    def _bound_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v


class LLMAdapter:
    prompt_header = (
        "You are an expert crypto futures trader. "
        "Respond ONLY with a single JSON object containing fields action, confidence, reason. "
        "Do not include any additional text. action must be LONG, SHORT or FLAT. "
        "Do not propose size, leverage, stop loss, take profit or prices."
    )

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def build_prompt(self, context: Dict[str, Any]) -> str:
        market_context = json.dumps(context, default=str)
        return f"{self.prompt_header}\nContext: {market_context}\nJSON:"

    def parse_output(self, content: str) -> LLMSignal:
        try:
            parsed = json.loads(content)
            response = LLMResponse(**parsed)
            return LLMSignal(action=response.action, confidence=response.confidence, reason=response.reason)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning("LLM output invalid", extra={"error": str(exc), "content": content})
            return LLMSignal(action=Side.FLAT, confidence=0.0, reason="invalid output")

    def infer(self, context: Dict[str, Any]) -> LLMSignal:
        if self.llm_client is None:
            return LLMSignal(action=Side.FLAT, confidence=0.0, reason="LLM disabled")

        prompt = self.build_prompt(context)
        try:
            raw = self.llm_client(prompt)
        except Exception as exc:  # pragma: no cover - external dependency
            logger.exception("LLM call failed", extra={"error": str(exc)})
            return LLMSignal(action=Side.FLAT, confidence=0.0, reason="LLM call failed")

        return self.parse_output(raw)
