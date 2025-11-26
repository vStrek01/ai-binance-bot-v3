"""Portfolio selection helpers for dry-run portfolio mode."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

from bot.core.config import BotConfig
from bot.utils.logger import get_logger

logger = get_logger(__name__)


ResultEntry = Dict[str, Any]


def _load_results(path: Path) -> List[ResultEntry]:
    if not path.exists():
        logger.debug("Optimization results file %s not found", path)
        return []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Optimization results file %s is corrupted", path)
        return []
    if isinstance(payload, list):
        return [cast(ResultEntry, entry) for entry in payload if isinstance(entry, dict)]
    logger.warning("Unexpected optimizer payload format in %s", path)
    return []


def load_optimizer_results(cfg: BotConfig, *, path: Optional[Path] = None) -> List[ResultEntry]:
    """Load flattened optimizer output if available."""
    target = path or (cfg.paths.optimization_dir / "best_params.json")
    return _load_results(target)


def select_top_markets(
    cfg: BotConfig,
    timeframe: str,
    top_n: int,
    metric: str,
    *,
    path: Optional[Path] = None,
) -> List[ResultEntry]:
    """Return optimizer entries ranked by the requested metric."""
    results = load_optimizer_results(cfg, path=path)
    if not results:
        return []
    filtered: List[ResultEntry] = []
    for entry in results:
        if entry.get("timeframe") != timeframe:
            continue
        metrics = cast(Dict[str, float], entry.get("metrics") or {})
        metric_value = metrics.get(metric)
        if metric_value is None:
            metric_value = metrics.get("total_pnl", 0.0)
        entry_copy = dict(entry)
        entry_copy["_rank_metric"] = float(metric_value)
        filtered.append(entry_copy)
    filtered.sort(key=lambda item: item.get("_rank_metric", 0.0), reverse=True)
    return filtered[:top_n]


def build_portfolio_meta(selection: Sequence[ResultEntry], metric: str) -> Dict[str, Any]:
    symbols = []
    for entry in selection:
        metrics = cast(Dict[str, float], entry.get("metrics") or {})
        symbols.append(
            {
                "symbol": entry.get("symbol"),
                "timeframe": entry.get("timeframe"),
                "metric": entry.get("_rank_metric"),
                "total_pnl": metrics.get("total_pnl"),
                "profit_factor": metrics.get("profit_factor"),
                "win_rate": metrics.get("win_rate"),
            }
        )
    return {
        "label": f"TOP {len(symbols)} ({metric})" if symbols else "EMPTY",
        "metric": metric,
        "symbols": symbols,
        "timeframe": selection[0]["timeframe"] if symbols else None,
    }


__all__ = ["select_top_markets", "build_portfolio_meta", "load_optimizer_results"]
