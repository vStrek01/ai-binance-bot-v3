from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Set

from bot.notifications.discord import DiscordNotifier

from .log_utils import DEFAULT_LOG_PATH, LogEvent, follow_log_events

DEFAULT_EVENTS = {"RISK_REJECTION", "SAFETY_HALT", "POSITION_LIQUIDATION"}


def format_alert(event: LogEvent) -> str:
    bits = [event.event]
    if event.symbol:
        bits.append(f"symbol={event.symbol}")
    reason = event.payload.get("reason") or event.payload.get("message")
    if reason:
        bits.append(str(reason))
    return " | ".join(bits)


def stream_alerts(
    log_path: Path,
    notifier: Optional[DiscordNotifier],
    *,
    events: Set[str],
    symbol: Optional[str],
    min_interval: float,
    dry_run: bool,
) -> None:
    last_sent = 0.0
    for event in follow_log_events(log_path, seek_end=True):
        if symbol and event.symbol != symbol:
            continue
        if events and event.event not in events:
            continue
        now = time.time()
        if now - last_sent < min_interval:
            continue
        message = format_alert(event)
        if dry_run:
            print(f"DRY RUN -> {message}")
        else:
            assert notifier is not None
            notifier.send(message, level="ALERT")
        last_sent = now


def main() -> None:
    parser = argparse.ArgumentParser(description="Push high-priority structured log events to Discord")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH, help="Structured bot log file")
    parser.add_argument("--webhook-url", default=os.getenv("DISCORD_WEBHOOK_URL"), help="Discord webhook URL or env var")
    parser.add_argument("--events", nargs="*", default=None, help="Event names to monitor (default: risk + safety)")
    parser.add_argument("--symbol", help="Optional symbol filter")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Minimum seconds between alerts")
    parser.add_argument("--dry-run", action="store_true", help="Print alerts instead of sending")
    parser.add_argument("--send-test", help="Send a single test message and exit")
    args = parser.parse_args()

    if not args.webhook_url and not args.dry_run:
        raise SystemExit("Missing webhook URL. Provide --webhook-url or set DISCORD_WEBHOOK_URL")

    notifier = DiscordNotifier(args.webhook_url) if not args.dry_run else None

    if args.send_test:
        if args.dry_run:
            print(f"TEST (dry run): {args.send_test}")
        elif notifier:
            notifier.send(args.send_test, level="TEST")
        return

    watch_events = set(args.events) if args.events else DEFAULT_EVENTS
    stream_alerts(
        args.log_file,
        notifier,
        events=watch_events,
        symbol=args.symbol,
        min_interval=args.cooldown,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
