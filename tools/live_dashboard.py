from __future__ import annotations

import argparse
import curses
import queue
import signal
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Optional

from .log_utils import DEFAULT_LOG_PATH, LogEvent, follow_log_events, iter_log_events

MAX_RECENT_EVENTS = 10


class DashboardState:
    def __init__(self, *, symbol_filter: Optional[str]) -> None:
        self.symbol_filter = symbol_filter
        self.equity: Optional[float] = None
        self.unrealized_pnl: Optional[float] = None
        self.positions: Dict[str, float] = {}
        self.last_signal: Dict[str, str] = {}
        self.last_order: Dict[str, str] = {}
        self.recent_events: Deque[str] = deque(maxlen=MAX_RECENT_EVENTS)
        self.last_event_ts: Optional[str] = None

    def apply(self, event: LogEvent) -> None:
        if self.symbol_filter and event.symbol and event.symbol != self.symbol_filter:
            return
        self.last_event_ts = event.recorded_ts
        if event.event == "EQUITY_SNAPSHOT":
            equity = event.payload.get("equity") or event.payload.get("balance")
            upnl = event.payload.get("unrealized_pnl") or event.payload.get("u_pnl")
            self.equity = _safe_float(equity)
            self.unrealized_pnl = _safe_float(upnl)
            self.recent_events.appendleft(f"Equity {self.equity} (uPnL {self.unrealized_pnl})")
        elif event.event == "STRATEGY_SIGNAL":
            summary = f"{event.payload.get('strategy')} -> {event.payload.get('signal')}"
            if event.symbol:
                self.last_signal[event.symbol] = summary
            self.recent_events.appendleft(f"Signal {event.symbol}: {summary}")
        elif event.event in {"ORDER_PLACED", "ORDER_FILLED", "ORDER_CANCELLED"}:
            qty = event.payload.get("quantity") or event.payload.get("qty")
            side = event.payload.get("side")
            price = event.payload.get("price")
            summary = f"{event.event.split('_')[1]} {side} {qty} @ {price}"
            if event.symbol:
                self.last_order[event.symbol] = summary
            self.recent_events.appendleft(f"Order {event.symbol}: {summary}")
        elif event.event == "POSITION_UPDATE":
            if event.symbol:
                size = _safe_float(event.payload.get("position"))
                if size is not None:
                    self.positions[event.symbol] = size
                    self.recent_events.appendleft(f"Pos {event.symbol}: {size}")
        elif event.event == "RISK_REJECTION":
            reason = event.payload.get("reason") or event.payload.get("message")
            self.recent_events.appendleft(f"Risk block ({event.symbol}): {reason}")


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _consumer(path: Path, event_queue: "queue.Queue[Optional[LogEvent]]", *, replay: bool) -> None:
    try:
        source = iter_log_events(path) if replay else follow_log_events(path, seek_end=True)
        for event in source:
            event_queue.put(event)
        event_queue.put(None)
    except Exception:
        event_queue.put(None)


def draw_screen(stdscr: "curses.window", state: DashboardState, *, start_time: float) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    title = f"AI Binance Bot Dashboard | {now}"
    stdscr.addstr(0, 0, title[: width - 1], curses.A_BOLD)

    info_lines = [
        f"Equity: {state.equity or 'n/a'}",
        f"Unrealized PnL: {state.unrealized_pnl or 'n/a'}",
        f"Last event: {state.last_event_ts or 'n/a'}",
        f"Uptime: {time.time() - start_time:0.0f}s",
    ]
    for idx, line in enumerate(info_lines, start=2):
        stdscr.addstr(idx, 0, line[: width - 1])

    stdscr.addstr(7, 0, "Signals:", curses.A_UNDERLINE)
    for row_idx, (symbol, summary) in enumerate(sorted(state.last_signal.items())[: height - 10]):
        stdscr.addstr(8 + row_idx, 0, f"{symbol}: {summary}"[: width - 1])

    col = width // 2
    stdscr.addstr(7, col, "Orders:", curses.A_UNDERLINE)
    for row_idx, (symbol, summary) in enumerate(sorted(state.last_order.items())[: height - 10]):
        stdscr.addstr(8 + row_idx, col, f"{symbol}: {summary}"[: width - 1])

    recent_start = height - MAX_RECENT_EVENTS - 2
    if recent_start > 8:
        stdscr.addstr(recent_start, 0, "Recent Events:", curses.A_UNDERLINE)
        for offset, entry in enumerate(list(state.recent_events)[: height - recent_start - 1]):
            stdscr.addstr(recent_start + 1 + offset, 0, entry[: width - 1])

    stdscr.refresh()


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime curses dashboard fed by structured bot logs.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--symbol", help="Optional symbol filter")
    parser.add_argument("--replay", action="store_true", help="Replay the full log instead of tailing live")
    parser.add_argument("--refresh", type=float, default=0.5, help="UI refresh interval in seconds")
    args = parser.parse_args()

    event_queue: "queue.Queue[Optional[LogEvent]]" = queue.Queue(maxsize=1000)
    worker = threading.Thread(target=_consumer, args=(args.log_file, event_queue), kwargs={"replay": args.replay}, daemon=True)
    worker.start()

    state = DashboardState(symbol_filter=args.symbol)
    start_time = time.time()

    def handle_interrupt(signum, frame):  # noqa: ANN001
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, handle_interrupt)

    def curses_loop(stdscr):  # noqa: ANN001
        curses.curs_set(0)
        stdscr.nodelay(True)
        while True:
            try:
                event = event_queue.get(timeout=args.refresh)
            except queue.Empty:
                event = None
            if event is None:
                if args.replay:
                    break
            else:
                state.apply(event)
            draw_screen(stdscr, state, start_time=start_time)

    try:
        curses.wrapper(curses_loop)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
