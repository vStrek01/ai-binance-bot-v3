"""Helpers for loading optimized parameter sets for strategies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

from bot.core.config import BotConfig
from bot.utils.logger import get_logger

logger = get_logger(__name__)


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Unable to parse %s: %s", path, exc)
        return None


ParamDict = Dict[str, float]


def load_best_params(cfg: BotConfig, symbol: str, interval: str) -> Optional[ParamDict]:
    """Load optimized parameters for a market if available."""
    base = cfg.paths.optimization_dir
    symbol_key = symbol.upper()
    interval_key = interval.lower()

    map_path = base / "best_params_by_market.json"
    payload = _read_json(map_path)
    if isinstance(payload, dict):
        params = _from_market_map(cast(Dict[str, Any], payload), symbol_key, interval_key)
        if params:
            return params

    best_path = base / "best_params.json"
    payload = _read_json(best_path)
    if isinstance(payload, list):
        params = _from_best_list(payload, symbol_key, interval_key)
        if params:
            return params
    return None


def _from_market_map(payload: Dict[str, Any], symbol: str, interval: str) -> Optional[ParamDict]:
    market = payload.get(symbol) or payload.get(symbol.upper()) or payload.get(symbol.lower())
    if not isinstance(market, dict):
        return None
    entry = market.get(interval) or market.get(interval.lower())
    if not isinstance(entry, dict):
        return None
    params = entry.get("params")
    return cast(Optional[ParamDict], params) if isinstance(params, dict) else None


def _from_best_list(payload: Any, symbol: str, interval: str) -> Optional[ParamDict]:
    matches: list[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        symbol_value = str(item.get("symbol", "")).upper()
        timeframe_value = item.get("timeframe")
        if symbol_value == symbol and timeframe_value == interval:
            matches.append(item)
    if not matches:
        return None
    matches.sort(key=lambda item: item.get("metrics", {}).get("total_pnl", float("-inf")), reverse=True)
    params = matches[0].get("params")
    return cast(Optional[ParamDict], params) if isinstance(params, dict) else None


__all__ = ["load_best_params"]
