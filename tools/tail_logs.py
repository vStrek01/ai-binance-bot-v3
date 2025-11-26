from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Literal, Optional, Set, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import infra.logging as logging_utils

DEFAULT_URL = "http://127.0.0.1:8000/api/dashboard/state"
LOCAL_MODE = "local"
API_MODE = "api"
MAX_EVENT_LIMIT = 200


def _clamp_limit(limit: int) -> int:
    if limit <= 0:
        return 1
    return min(limit, MAX_EVENT_LIMIT)


def _with_query(url: str, events_limit: int) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}limit={events_limit}&t={int(time.time() * 1000)}"


def _fetch_state(url: str, events_limit: int) -> Dict[str, Any]:
    request = Request(_with_query(url, _clamp_limit(events_limit)), headers={"Accept": "application/json"})
    with urlopen(request, timeout=5) as response:  # nosec - local-only monitor
        return json.loads(response.read().decode("utf-8"))


def _local_state(limit: int) -> Dict[str, Any]:
    sanitized = _clamp_limit(limit)
    return {
        "equity": logging_utils.get_equity_snapshot(),
        "positions": logging_utils.get_open_positions(),
        "recent_events": logging_utils.get_recent_events(limit=sanitized),
        "limit": sanitized,
    }


def _format_currency(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0.00"
    return f"{number:,.2f}"


def _event_key(payload: Dict[str, Any]) -> str:
    parts = [
        payload.get("timestamp"),
        payload.get("event"),
        payload.get("symbol"),
        payload.get("side"),
        payload.get("status"),
    ]
    return "::".join(str(part) for part in parts)


def _position_key(position: Dict[str, Any]) -> str:
    return "::".join(
        str(position.get(part, ""))
        for part in ("symbol", "side", "entry_price", "quantity", "stop_loss", "take_profit")
    )


def _remember(cache: Deque[str], seen: Set[str], key: str) -> bool:
    if key in seen:
        return False
    if len(cache) == cache.maxlen:
        dropped = cache.popleft()
        seen.discard(dropped)
    cache.append(key)
    seen.add(key)
    return True


def _print_positions(positions: Iterable[Dict[str, Any]]) -> None:
    for pos in positions:
        print(
            "[POS]",
            pos.get("symbol", "?"),
            pos.get("side", "?"),
            f"qty={_format_currency(pos.get('quantity'))}",
            f"entry={_format_currency(pos.get('entry_price'))}",
            f"mark={_format_currency(pos.get('mark_price'))}",
            flush=True,
        )


def tail(url: str, interval: float, events_limit: int, buffer_size: int, source: Literal["api", "local"] = API_MODE) -> None:
    seen_events: Deque[str] = deque(maxlen=buffer_size)
    seen_lookup: Set[str] = set()
    last_equity_stamp: Optional[str] = None
    last_position_keys: Set[str] = set()

    while True:
        try:
            if source == API_MODE:
                state = cast(Dict[str, Any], _fetch_state(url, events_limit))
            else:
                state = _local_state(events_limit)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[tail] request failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(interval)
            continue
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"[tail] unexpected error: {exc}", file=sys.stderr, flush=True)
            time.sleep(interval)
            continue

        equity = cast(Dict[str, Any], state.get("equity") or {})
        stamp = equity.get("timestamp") or state.get("updated_at")
        if stamp and stamp != last_equity_stamp:
            print(
                "[EQUITY]",
                stamp,
                f"equity={_format_currency(equity.get('equity'))}",
                f"balance={_format_currency(equity.get('balance'))}",
                f"open_positions={equity.get('open_positions', 0)}",
                flush=True,
            )
            last_equity_stamp = stamp

        positions = cast(List[Dict[str, Any]], list(state.get("positions") or []))
        position_keys = {_position_key(pos) for pos in positions}
        if position_keys != last_position_keys:
            print(f"[STATE] {len(positions)} open positions", flush=True)
            _print_positions(positions)
            last_position_keys = position_keys

        recent_events = cast(List[Dict[str, Any]], list(state.get("recent_events", [])))
        pending: List[Dict[str, Any]] = []
        for event in reversed(recent_events):  # newest entries are first in the payload
            if _remember(seen_events, seen_lookup, _event_key(event)):
                pending.append(event)
        for event in pending:
            label = event.get("event", "?")
            symbol = event.get("symbol")
            side = event.get("side")
            price = event.get("fill_price") or event.get("entry_price")
            qty = event.get("qty") or event.get("quantity")
            print(
                "[EVENT]",
                event.get("timestamp", ""),
                label,
                symbol or "",
                side or "",
                f"qty={_format_currency(qty)}" if qty is not None else "",
                f"price={_format_currency(price)}" if price is not None else "",
                f"reason={event.get('reason', '')}" if event.get("reason") else "",
                flush=True,
            )

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail the local dashboard API for recent events and equity snapshots")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dashboard state endpoint (default: %(default)s)")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--events-limit", type=int, default=30, help="Maximum events to request per poll")
    parser.add_argument("--buffer-size", type=int, default=200, help="How many event IDs to keep for deduplication")
    parser.add_argument(
        "--source",
        choices=(API_MODE, LOCAL_MODE),
        default=API_MODE,
        help="Use API polling (default) or read directly from in-process logging helpers",
    )
    args = parser.parse_args()
    try:
        tail(args.url, args.interval, args.events_limit, args.buffer_size, source=args.source)
    except KeyboardInterrupt:
        print("\n[tail] stopped", flush=True)


if __name__ == "__main__":
    main()
