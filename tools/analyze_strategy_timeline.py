from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from .log_utils import DEFAULT_LOG_PATH, LogEvent, coerce_timestamp, iter_log_events

INTERESTING_EVENTS = {
    "STRATEGY_SIGNAL",
    "ORDER_PLACED",
    "ORDER_FILLED",
    "ORDER_CANCELLED",
    "RISK_REJECTION",
    "LEVERAGE_UPDATE",
    "EQUITY_SNAPSHOT",
}

def describe_event(event: LogEvent) -> str:
    payload = event.payload
    event_type = event.event
    if event_type == "STRATEGY_SIGNAL":
        strategy = payload.get("strategy")
        direction = payload.get("signal") or payload.get("direction")
        price = payload.get("price")
        return f"{strategy or '?'} {direction or '?'} @ {price}"
    if event_type.startswith("ORDER_"):
        side = payload.get("side")
        qty = payload.get("quantity") or payload.get("qty")
        price = payload.get("price")
        order_id = payload.get("order_id") or payload.get("client_order_id")
        return f"{side or '?'} {qty} @ {price} (id={order_id})"
    if event_type == "RISK_REJECTION":
        reason = payload.get("reason") or payload.get("message")
        return f"Rejected: {reason}"
    if event_type == "LEVERAGE_UPDATE":
        leverage = payload.get("leverage")
        mode = payload.get("mode")
        return f"Leverage -> {leverage} ({mode})"
    if event_type == "EQUITY_SNAPSHOT":
        equity = payload.get("equity") or payload.get("balance")
        upnl = payload.get("unrealized_pnl")
        return f"Equity {equity} (uPnL={upnl})"
    return ""


def build_rows(
    events: Iterable[LogEvent],
    *,
    symbol: Optional[str],
    lookback_minutes: int,
) -> List[List[str]]:
    cutoff = None
    now = datetime.now(timezone.utc)
    if lookback_minutes > 0:
        cutoff = now - timedelta(minutes=lookback_minutes)
    rows: List[List[str]] = []
    for event in events:
        if event.event not in INTERESTING_EVENTS:
            continue
        if symbol and event.symbol and event.symbol != symbol:
            continue
        ts = coerce_timestamp(event.recorded_ts)
        if cutoff and ts and ts < cutoff:
            continue
        display_ts = ts.isoformat(timespec="seconds") if ts else event.recorded_ts
        details = describe_event(event)
        rows.append([display_ts, event.event, event.symbol or "-", details])
    return rows


def format_table(rows: List[List[str]]) -> str:
    headers = ["Timestamp", "Event", "Symbol", "Details"]
    data = [headers] + rows
    col_widths = [max(len(str(row[idx])) for row in data) for idx in range(len(headers))]
    lines: List[str] = []
    for i, row in enumerate(data):
        padded = [str(cell).ljust(col_widths[idx]) for idx, cell in enumerate(row)]
        lines.append(" | ".join(padded))
        if i == 0:
            lines.append("-+-".join("-" * width for width in col_widths))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect recent structured strategy events.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH, help="Path to bot.log (defaults to logs/bot.log)")
    parser.add_argument("--symbol", help="Filter for one symbol (optional)")
    parser.add_argument("--lookback-minutes", type=int, default=120, help="Number of minutes to include (default: 120)")
    parser.add_argument("--limit", type=int, default=200, help="Max rows to show (default: 200)")
    args = parser.parse_args()

    events = iter_log_events(args.log_file)
    rows = build_rows(events, symbol=args.symbol, lookback_minutes=args.lookback_minutes)
    if not rows:
        print("No matching events found.")
        return
    if args.limit > 0:
        rows = rows[-args.limit :]
    print(format_table(rows))


if __name__ == "__main__":
    main()
