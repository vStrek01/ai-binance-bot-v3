from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from infra.logging import log_event, logger


class StateStore:
    """Persist arbitrary bot state to disk for later recovery."""

    def __init__(self, run_id: str, base_dir: str | Path = "data/state") -> None:
        self.run_id = run_id
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / f"{self.run_id}.json"
        self._lock = Lock()
        self._cache: Dict[str, Any] = {}
        self._load_into_cache()

    def save(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")
        with self._lock:
            self._write(state)

    def load(self) -> Dict[str, Any]:
        with self._lock:
            self._load_into_cache()
            payload = dict(self._cache)
        log_event("state_loaded", run_id=self.run_id, sections=list(payload.keys()))
        return payload

    def merge(self, **sections: Dict[str, Any]) -> Dict[str, Any]:
        if not sections:
            return self.load()
        with self._lock:
            state = dict(self._cache)
            for key, value in sections.items():
                state[key] = value
            self._write(state, changed=list(sections.keys()))
            snapshot = dict(state)
        return snapshot

    # ------------------------------------------------------------------

    def _load_into_cache(self) -> None:
        if not self.path.exists():
            self._cache = {}
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            payload = json.loads(raw) if raw else {}
            if isinstance(payload, dict):
                self._cache = payload
                return
            logger.warning("State file payload is not a dict", extra={"path": str(self.path)})
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.warning("Unable to parse state file", extra={"path": str(self.path), "error": str(exc)})
        self._cache = {}

    def _write(self, state: Dict[str, Any], *, changed: list[str] | None = None) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        data = json.dumps(state, default=str, indent=2)
        tmp_path.write_text(data, encoding="utf-8")
        os.replace(tmp_path, self.path)
        self._cache = dict(state)
        log_event("state_saved", run_id=self.run_id, sections=list(self._cache.keys()), changed_sections=changed)
