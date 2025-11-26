"""Parameter search utility for the EMA/RSI/ATR strategy."""
from __future__ import annotations

import itertools
import json
import random
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bot import config
from bot.backtester import Backtester
from bot.data import load_local_candles
from bot.execution.client_factory import build_data_client
from bot.exchange_info import ExchangeInfoManager, SymbolFilters
from bot.status import status_store
from bot.strategies import build_parameters
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger

logger = get_logger(__name__)

ParamDict = Dict[str, float | int]
Snapshot = Dict[str, Dict[str, float]]
Task = Tuple[str, str, ParamDict, Snapshot]


def _normalize(params: ParamDict) -> ParamDict:
    normalized: ParamDict = {}
    defaults = config.strategy.default_parameters
    for key, value in params.items():
        template = defaults.get(key)
        if isinstance(template, int):
            normalized[key] = int(value)
        else:
            normalized[key] = float(value)
    return normalized


def _run_task(task: Task) -> Dict[str, Any]:
    symbol, timeframe, params, snapshot = task
    prefetched = {sym: SymbolFilters(**data) for sym, data in snapshot.items()}
    exchange = ExchangeInfoManager(client=None, prefetched=prefetched)
    backtester = Backtester(exchange)
    try:
        candles = load_local_candles(symbol, timeframe)
    except FileNotFoundError:
        return {"symbol": symbol, "timeframe": timeframe, "error": "missing_data"}
    normalized = _normalize(params)
    strategy_params = build_parameters(normalized)
    outcome = backtester.run(symbol, timeframe, candles, strategy_params)
    return {"symbol": symbol, "timeframe": timeframe, "params": normalized, "metrics": outcome["metrics"]}


class Optimizer:
    def __init__(
        self,
        symbols: Iterable[str],
        timeframes: Iterable[str],
        pairs: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> None:
        self.symbols = list(symbols)
        self.timeframes = list(timeframes)
        self._pairs = [(sym, tf) for sym, tf in pairs] if pairs else None
        client = build_data_client()
        self.exchange_info = ExchangeInfoManager(client=client)
        self.exchange_info.refresh()

    def _param_sets(self) -> List[ParamDict]:
        base = config.strategy.default_parameters.copy()
        space = config.strategy.parameter_space
        if not space:
            return [base]
        keys = list(space.keys())
        combos = list(itertools.product(*[space[key] for key in keys]))
        if config.optimizer.randomize:
            random.shuffle(combos)
        if config.optimizer.max_param_combinations is not None:
            combos = combos[: config.optimizer.max_param_combinations]
        results: List[ParamDict] = []
        for combo in combos:
            params = base.copy()
            for key, value in zip(keys, combo):
                params[key] = value
            results.append(params)
        return results or [base]

    def _limited_symbols(self) -> List[str]:
        symbols = self.symbols
        if config.optimizer.max_symbols_per_run is not None:
            symbols = symbols[: config.optimizer.max_symbols_per_run]
        return symbols

    def run(self) -> List[Dict[str, Any]]:
        param_sets = self._param_sets()
        snapshot = self.exchange_info.snapshot()
        markets = self._markets()
        tasks: List[Task] = [
            (symbol, timeframe, params, snapshot)
            for symbol, timeframe in markets
            for params in param_sets
        ]
        total = len(tasks)
        logger.info("Optimizer will evaluate %s combinations", total)
        if not tasks:
            return []

        results: List[Dict[str, Any]] = []
        completed = 0
        best_result: Optional[Dict[str, Any]] = None

        def handle_result(result: Dict[str, Any]) -> None:
            nonlocal completed, best_result
            results.append(result)
            completed += 1
            status_store.set_progress(completed, total)
            if "symbol" in result and "timeframe" in result:
                status_store.set_mode("optimize", result.get("symbol"), result.get("timeframe"))
            metrics = result.get("metrics")
            if not metrics:
                return
            if best_result is None or metrics.get("total_pnl", 0.0) > best_result["metrics"].get("total_pnl", 0.0):
                best_result = result
                status_store.set_metrics(metrics)

        status_store.set_mode("optimize", None, None)
        status_store.set_progress(0, total)
        try:
            if config.optimizer.enable_parallel and total > 1:
                workers = min(config.optimizer.max_workers or cpu_count(), cpu_count())
                with Pool(processes=workers) as pool:
                    for item in pool.imap_unordered(_run_task, tasks):
                        handle_result(item)
            else:
                for task in tasks:
                    handle_result(_run_task(task))
        finally:
            status_store.clear_progress()
            status_store.set_mode("idle", None, None)

        filtered = [result for result in results if "metrics" in result]
        filtered.sort(key=lambda item: item["metrics"]["total_pnl"], reverse=True)
        self._persist(filtered)
        return filtered

    def _markets(self) -> List[Tuple[str, str]]:
        if self._pairs is not None:
            allowed = set(self._limited_symbols()) if config.optimizer.max_symbols_per_run else None
            markets = [pair for pair in self._pairs if pair[1] in self.timeframes]
            if allowed is not None:
                markets = [pair for pair in markets if pair[0] in allowed]
            return markets
        return [(symbol, timeframe) for symbol in self._limited_symbols() for timeframe in self.timeframes]

    def _persist(self, results: List[Dict[str, Any]]) -> None:
        config.ensure_directories()
        config.OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)
        json_path = config.OPTIMIZATION_DIR / "best_params.json"
        csv_path = config.OPTIMIZATION_DIR / "best_params.csv"
        map_path = config.OPTIMIZATION_DIR / "best_params_by_market.json"

        json_payload = [
            {"symbol": r["symbol"], "timeframe": r["timeframe"], "params": r["params"], "metrics": r["metrics"]}
            for r in results
        ]
        atomic_write_text(json_path, json.dumps(json_payload, indent=2))

        best_map = self._best_map(results)
        atomic_write_text(map_path, json.dumps(best_map, indent=2))

        flat_rows = []
        for record in results:
            row: Dict[str, Any] = {
                "symbol": record["symbol"],
                "timeframe": record["timeframe"],
            }
            row.update({f"param_{k}": v for k, v in record["params"].items()})
            row.update({f"metric_{k}": v for k, v in record["metrics"].items()})
            flat_rows.append(row)
        if flat_rows:
            csv_payload = pd.DataFrame(flat_rows).to_csv(index=False)
            atomic_write_text(csv_path, csv_payload)
        else:
            if csv_path.exists():
                csv_path.unlink()
        logger.info("Optimizer results written to %s and %s", json_path, csv_path)

    @staticmethod
    def _best_map(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for record in results:
            symbol = record["symbol"]
            timeframe = record["timeframe"]
            metrics = record.get("metrics", {})
            best_for_symbol = grouped.setdefault(symbol, {})
            current = best_for_symbol.get(timeframe)
            if current is None or metrics.get("total_pnl", 0.0) > current["metrics"].get("total_pnl", 0.0):
                best_for_symbol[timeframe] = {"params": record["params"], "metrics": metrics}
        return grouped


def load_best_parameters(symbol: str, timeframe: str) -> Optional[ParamDict]:
    """Load the best saved parameters for a market, if available."""
    candidates = [
        config.OPTIMIZATION_DIR / "learned_params.json",
        config.OPTIMIZATION_DIR / "hyper_params.json",
        config.OPTIMIZATION_DIR / "best_params_by_market.json",
        config.OPTIMIZATION_DIR / "best_params.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        params = _extract_params_from_payload(payload, symbol, timeframe)
        if params:
            return _normalize(params)
    return None


def _extract_params_from_payload(payload: Any, symbol: str, timeframe: str) -> Optional[ParamDict]:
    if isinstance(payload, list):
        winners = [entry for entry in payload if entry.get("symbol") == symbol and entry.get("timeframe") == timeframe]
        if not winners:
            return None
        winners.sort(key=lambda item: item.get("metrics", {}).get("total_pnl", 0.0), reverse=True)
        return winners[0].get("params")
    if isinstance(payload, dict):
        market = payload.get(symbol, {})
        entry = market.get(timeframe)
        if entry:
            return entry.get("params")
    return None


__all__ = ["Optimizer", "load_best_parameters"]
