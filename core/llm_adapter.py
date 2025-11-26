from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ai.decision_schema import LlmDecision, parse_llm_response
from core.models import LLMSignal, Side
from infra.logging import logger, log_event


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

    def parse_output(self, content: str, *, meta: Optional[Dict[str, Any]] = None) -> LLMSignal:
        decision = parse_llm_response(content)
        if decision is None:
            self._log_invalid_output(content, meta)
            return LLMSignal(action=Side.FLAT, confidence=0.0, reason="invalid output")
        return self._decision_to_signal(decision)

    def infer(self, context: Dict[str, Any]) -> LLMSignal:
        if self.llm_client is None:
            return LLMSignal(action=Side.FLAT, confidence=0.0, reason="LLM disabled")

        prompt = self.build_prompt(context)
        try:
            raw = self.llm_client(prompt)
        except Exception as exc:  # pragma: no cover - external dependency
            logger.exception("LLM call failed", extra={"error": str(exc)})
            return LLMSignal(action=Side.FLAT, confidence=0.0, reason="LLM call failed")

        return self.parse_output(raw, meta=context)

    def _decision_to_signal(self, decision: LlmDecision) -> LLMSignal:
        confidence = float(decision.confidence) / 100.0
        return LLMSignal(action=Side(decision.action), confidence=confidence, reason=decision.reason)

    def _log_invalid_output(self, raw: Optional[str], meta: Optional[Dict[str, Any]] = None) -> None:
        snippet = (raw or "")[:200]
        details = {
            "truncated_raw": snippet,
        }
        if meta:
            for field in ("symbol", "interval", "run_mode"):
                if meta.get(field) is not None:
                    details[field] = meta[field]
        logger.warning("LLM_INVALID_OUTPUT", extra={"event": "LLM_INVALID_OUTPUT", "snippet": snippet})
        log_event("LLM_INVALID_OUTPUT", **details)
