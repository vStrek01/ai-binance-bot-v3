"""Smoke test to ensure FastAPI dashboard endpoints respond."""
from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from bot.api import app


def _assert_status_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("/api/status payload must be a JSON object")


def main() -> int:
    client = TestClient(app)
    status_resp = client.get("/api/status")
    status_resp.raise_for_status()
    _assert_status_payload(status_resp.json())

    reset_resp = client.post("/api/reset-stats")
    reset_resp.raise_for_status()
    reset_payload = reset_resp.json()
    if not isinstance(reset_payload, dict) or reset_payload.get("status") != "ok":
        raise ValueError("/api/reset-stats did not return status=ok")

    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Preflight API smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
