from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from tools.log_utils import DEFAULT_LOG_PATH, LogEvent, follow_log_events

ALERT_EVENTS = {"RISK_REJECTION", "SAFETY_HALT", "POSITION_LIQUIDATION"}
DEFAULT_USERNAME = "AI Binance Bot"


class DiscordNotifier:
    """Minimal Discord webhook client."""

    def __init__(self, webhook_url: str, *, username: str = DEFAULT_USERNAME, session: Optional[requests.Session] = None) -> None:
        self.webhook_url = webhook_url
        self.username = username
        self._session = session or requests.Session()

    def send(self, content: str, *, level: str = "INFO", embed: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "content": f"[{level.upper()}] {content}",
            "username": self.username,
        }
        if embed:
            payload["embeds"] = [embed]
        response = self._session.post(self.webhook_url, json=payload, timeout=10)
        if response.status_code >= 300:
            raise RuntimeError(f"Discord webhook failed: {response.status_code} {response.text}")


def format_event(event: LogEvent) -> str:
    base = f"{event.event} for {event.symbol or 'n/a'}"
    if event.event == "RISK_REJECTION":
        reason = event.payload.get("reason") or event.payload.get("message")
        return f"{base}: {reason}"
    if event.event == "SAFETY_HALT":
        return f"{base}: {event.payload.get('details') or 'Safety module halt triggered'}"
    return base


def stream_alerts(log_path: Path, notifier: DiscordNotifier, *, symbol: Optional[str]) -> None:
    for event in follow_log_events(log_path, seek_end=True):
        if event.event not in ALERT_EVENTS:
            continue
        if symbol and event.symbol != symbol:
            continue
        notifier.send(format_event(event), level="ALERT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send ad-hoc Discord alerts or stream high-priority log events.")
    parser.add_argument("--webhook-url", default=os.getenv("DISCORD_WEBHOOK_URL"), help="Discord webhook URL or DISCORD_WEBHOOK_URL env var")
    parser.add_argument("--message", help="Send this message once and exit")
    parser.add_argument("--level", default="INFO", help="Message level prefix")
    parser.add_argument(
        "--watch-log",
        type=Path,
        nargs="?",
        const=DEFAULT_LOG_PATH,
        help="Tail logs (default logs/bot.log) and push risk alerts",
    )
    parser.add_argument("--symbol", help="Optional symbol filter for watch mode")
    args = parser.parse_args()

    if not args.webhook_url:
        raise SystemExit("Webhook URL missing. Pass --webhook-url or set DISCORD_WEBHOOK_URL")

    notifier = DiscordNotifier(args.webhook_url)

    if args.watch_log:
        stream_alerts(args.watch_log, notifier, symbol=args.symbol)
        return

    if args.message:
        notifier.send(args.message, level=args.level)
        return

    parser.error("Either --message or --watch-log must be provided")


if __name__ == "__main__":
    main()
