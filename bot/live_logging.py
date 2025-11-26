"""Utilities for persisting live trading activity."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from bot import config


class OrderAuditLogger:
    """Append-only JSONL logger for order requests and responses."""

    def __init__(self, path: Path | None = None) -> None:
        config.ensure_directories([config.LOG_DIR])
        self.path = path or (config.LOG_DIR / "live_orders.log")

    def log(self, payload: Dict[str, Any]) -> None:
        record = json.dumps(payload, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")


class LiveTradeLogger:
    """Write each executed live trade to CSV and JSONL for later analysis."""

    CSV_HEADERS: Iterable[str] = (
        "mode",
        "symbol",
        "timeframe",
        "side",
        "quantity",
        "entry_price",
        "exit_price",
        "pnl",
        "opened_at",
        "closed_at",
        "reason",
        "entry_order_id",
        "exit_order_id",
    )

    def __init__(self, csv_path: Path | None = None, jsonl_path: Path | None = None) -> None:
        config.ensure_directories([config.LOG_DIR])
        self.csv_path = csv_path or (config.LOG_DIR / "live_trades.csv")
        self.jsonl_path = jsonl_path or (config.LOG_DIR / "live_trades.jsonl")

    def log(self, trade: Dict[str, Any]) -> None:
        self._append_jsonl(trade)
        self._append_csv(trade)

    def _append_jsonl(self, trade: Dict[str, Any]) -> None:
        record = json.dumps(trade, default=str)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(record + "\n")

    def _append_csv(self, trade: Dict[str, Any]) -> None:
        file_exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.CSV_HEADERS))
            if not file_exists:
                writer.writeheader()
            row = {key: trade.get(key) for key in self.CSV_HEADERS}
            writer.writerow(row)


__all__ = ["OrderAuditLogger", "LiveTradeLogger"]
