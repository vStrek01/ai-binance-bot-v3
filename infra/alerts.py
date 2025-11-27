"""Lightweight alert dispatch helper with pluggable transports."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from urllib import error, request

AlertTransport = Callable[[str, bytes, Dict[str, str], float], None]


@dataclass(frozen=True)
class AlertTargets:
    webhooks: List[str]
    telegram: List[Dict[str, str]]


_alert_transport: AlertTransport | None = None
_TRANSPORT_LOCK = threading.Lock()
_DEFAULT_TIMEOUT = float(os.getenv("ALERT_TIMEOUT", "5"))


def set_transport(transport: Optional[AlertTransport]) -> None:
    """Override the default HTTP transport (used by tests)."""

    global _alert_transport
    with _TRANSPORT_LOCK:
        _alert_transport = transport


def _get_transport() -> AlertTransport:
    with _TRANSPORT_LOCK:
        if _alert_transport is not None:
            return _alert_transport
    return _default_transport


def _default_transport(url: str, body: bytes, headers: Dict[str, str], timeout: float) -> None:
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout):
            return
    except error.URLError:
        return


def _resolve_targets() -> AlertTargets:
    webhook = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    telegram_targets: List[Dict[str, str]] = []
    if token and chat_id:
        telegram_targets.append({"token": token, "chat_id": chat_id})
    webhooks = [webhook] if webhook else []
    return AlertTargets(webhooks=webhooks, telegram=telegram_targets)


def send_alert(event: str, *, severity: str = "warning", message: Optional[str] = None, **fields: Any) -> bool:
    targets = _resolve_targets()
    if not targets.webhooks and not targets.telegram:
        return False
    payload: Dict[str, Any] = {
        "event": event,
        "severity": severity,
        "message": message or event.replace("_", " "),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": fields,
    }
    transport = _get_transport()
    dispatched = False
    body = json.dumps(payload, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    for url in targets.webhooks:
        try:
            transport(url, body, headers, _DEFAULT_TIMEOUT)
            dispatched = True
        except Exception:  # noqa: BLE001 - alerts must not interrupt trading
            continue
    for target in targets.telegram:
        telegram_url = f"https://api.telegram.org/bot{target['token']}/sendMessage"
        text_payload: Dict[str, Any] = {
            "chat_id": target["chat_id"],
            "text": _render_telegram_message(payload),
            "disable_web_page_preview": True,
        }
        try:
            transport(
                telegram_url,
                json.dumps(text_payload, default=str).encode("utf-8"),
                headers,
                _DEFAULT_TIMEOUT,
            )
            dispatched = True
        except Exception:  # noqa: BLE001 - non-fatal
            continue
    return dispatched


def _render_telegram_message(payload: Dict[str, Any]) -> str:
    event = payload.get("event", "ALERT")
    severity = payload.get("severity", "warning")
    message = payload.get("message", event)
    details = payload.get("details")
    text = f"[{severity}] {event}: {message}"
    if isinstance(details, dict) and details:
        detail_parts = []
        for key, value in details.items():
            if value is None:
                continue
            detail_parts.append(f"{key}={value}")
        if detail_parts:
            text += "\n" + " ".join(detail_parts)
    return text


__all__ = ["send_alert", "set_transport", "AlertTargets"]
