"""Tail STRATEGY_* events from the bot log for demo-live monitoring."""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

INTERESTING_EVENTS = {"STRATEGY_TICK", "STRATEGY_VETO", "STRATEGY_DECISION"}
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "bot.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow STRATEGY_* events in near real time.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the log file (defaults to logs/bot.log).",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=200,
        help="How many matching lines to print from history before tailing (default: 200).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Polling interval in seconds when waiting for new log lines (default: 1.5).",
    )
    return parser.parse_args()


def wait_for_file(path: Path) -> None:
    while not path.exists():
        print(f"Waiting for {path} ...", end="\r", flush=True)
        time.sleep(1.0)
    print("")


def format_line(line: str) -> Optional[str]:
    brace = line.find("{")
    if brace == -1:
        return None
    payload = line[brace:]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    event = data.get("event")
    if event not in INTERESTING_EVENTS:
        return None
    timestamp = line[:brace].strip()
    trimmed = {k: v for k, v in data.items() if k != "event"}
    formatted_payload = json.dumps(trimmed, separators=(",", ":"), ensure_ascii=False)
    return f"{timestamp} {event} {formatted_payload}".strip()


def print_recent(path: Path, limit: int) -> None:
    recent: Deque[str] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                formatted = format_line(raw)
                if formatted:
                    recent.append(formatted)
    except FileNotFoundError:
        return
    if not recent:
        print("No STRATEGY_* entries yet.")
        return
    print(f"--- Last {len(recent)} STRATEGY_* entries from {path} ---")
    for entry in recent:
        print(entry)
    print("--- Streaming live updates (Ctrl+C to exit) ---")


def tail_log(path: Path, interval: float) -> None:
    while True:
        wait_for_file(path)
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(0, 2)
                while True:
                    line = handle.readline()
                    if not line:
                        time.sleep(interval)
                        if not path.exists():
                            break
                        continue
                    formatted = format_line(line)
                    if formatted:
                        print(formatted, flush=True)
        except FileNotFoundError:
            pass
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging helper
            print(f"Watcher encountered an error: {exc}")
            time.sleep(interval)


def main() -> None:
    args = parse_args()
    log_path = args.log_file.resolve()
    print_recent(log_path, max(1, args.lines))
    try:
        tail_log(log_path, max(0.5, args.interval))
    except KeyboardInterrupt:
        print("\nStopped log tail.")


if __name__ == "__main__":
    main()
