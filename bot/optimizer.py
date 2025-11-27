"""Parameter search utility for the EMA/RSI/ATR strategy."""
# EXPERIMENTAL — DO NOT ENABLE FOR LIVE TRADING WITHOUT SEPARATE VALIDATION.
from __future__ import annotations

import itertools
import json
import random
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bot.backtester import Backtester
from bot.data import load_local_candles
from bot.execution.client_factory import build_data_client
from bot.exchange_info import ExchangeInfoManager, SymbolFilters
from bot.core.config import BotConfig, ensure_directories
from bot.status import status_store
from bot.strategies import build_parameters
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger
from exchange.symbols import SymbolInfo

logger = get_logger(__name__)

ParamDict = Dict[str, float | int]
Snapshot = Dict[str, Dict[str, Any]]
Task = Tuple[str, str, ParamDict, Snapshot]


_OPTIMIZER_CFG: BotConfig | None = None


def _set_optimizer_config(cfg: BotConfig) -> None:
    global _OPTIMIZER_CFG
    _OPTIMIZER_CFG = cfg


def _require_optimizer_config() -> BotConfig:
    if _OPTIMIZER_CFG is None:
        raise RuntimeError("Optimizer config not initialized")
    return _OPTIMIZER_CFG


def _normalize(params: ParamDict, cfg: BotConfig) -> ParamDict:
    normalized: ParamDict = {}
    defaults = cfg.strategy.default_parameters
    for key, value in params.items():
        template = defaults.get(key)
        if isinstance(template, int):
            normalized[key] = int(value)
        else:
            normalized[key] = float(value)
    return normalized


def _filters_from_snapshot(data: Dict[str, Any]) -> SymbolFilters:
    payload = data.get("symbol_info") if isinstance(data, dict) else None
    if payload:
        info = SymbolInfo.from_dict(payload)
        return SymbolFilters.from_symbol_info(info)
    sanitized = {k: v for k, v in data.items() if k != "symbol_info"}
    return SymbolFilters(**sanitized)


def _run_task(task: Task) -> Dict[str, Any]:
    cfg = _require_optimizer_config()
    symbol, timeframe, params, snapshot = task
    prefetched = {sym: _filters_from_snapshot(data) for sym, data in snapshot.items()}
    exchange = ExchangeInfoManager(cfg, client=None, prefetched=prefetched)
    backtester = Backtester(cfg, exchange)
    try:
        candles = load_local_candles(cfg, symbol, timeframe)
    except FileNotFoundError:
        return {"symbol": symbol, "timeframe": timeframe, "error": "missing_data"}
    normalized = _normalize(params, cfg)
    strategy_params = build_parameters(cfg, symbol=symbol, overrides=normalized)
    outcome = backtester.run(symbol, timeframe, candles, strategy_params)
    return {"symbol": symbol, "timeframe": timeframe, "params": normalized, "metrics": outcome["metrics"]}


class Optimizer:
    def __init__(
        self,
        symbols: Iterable[str],
        timeframes: Iterable[str],
        pairs: Optional[Iterable[Tuple[str, str]]] = None,
        cfg: BotConfig | None = None,
    ) -> None:
        if cfg is None:
            raise ValueError("Optimizer requires a BotConfig instance")
        self._config = cfg
        self.symbols = list(symbols)
        self.timeframes = list(timeframes)
        self._pairs = [(sym, tf) for sym, tf in pairs] if pairs else None
        client = build_data_client(cfg)
        self.exchange_info = ExchangeInfoManager(cfg, client=client)
        self.exchange_info.refresh()

    def _param_sets(self) -> List[ParamDict]:
        base = self._config.strategy.default_parameters.copy()
        space = self._config.strategy.parameter_space
        if not space:
            return [base]
        keys = list(space.keys())
        combos = list(itertools.product(*[space[key] for key in keys]))
        rng = random.Random(self._config.optimizer.random_seed)
        search_mode = self._config.optimizer.search_mode
        if search_mode == "random":
            target = (
                self._config.optimizer.random_subset
                or self._config.optimizer.max_param_combinations
                or len(combos)
            )
            sample_size = min(len(combos), max(1, target))
            if sample_size < len(combos):
                combos = rng.sample(combos, sample_size)
            else:
                rng.shuffle(combos)
        else:
            if self._config.optimizer.randomize:
                rng.shuffle(combos)
            if self._config.optimizer.max_param_combinations is not None:
                combos = combos[: self._config.optimizer.max_param_combinations]
        results: List[ParamDict] = []
        for combo in combos:
            params = base.copy()
            for key, value in zip(keys, combo):
                params[key] = value
            results.append(params)
        return results or [base]

    def _limited_symbols(self) -> List[str]:
        symbols = self.symbols
        if self._config.optimizer.max_symbols_per_run is not None:
            symbols = symbols[: self._config.optimizer.max_symbols_per_run]
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

        _set_optimizer_config(self._config)

        results: List[Dict[str, Any]] = []
        completed = 0
        best_result: Optional[Dict[str, Any]] = None
        best_score: Optional[float] = None
        score_label = self._config.optimizer.score_metric or "total_pnl"
        min_improvement = max(self._config.optimizer.min_improvement, 0.0)
        patience = self._config.optimizer.early_stop_patience
        allow_early_stop = (
            not self._config.optimizer.enable_parallel
            and patience is not None
            and patience > 0
        )
        no_improve = 0

        def handle_result(result: Dict[str, Any]) -> bool:
            nonlocal completed, best_result, best_score
            results.append(result)
            completed += 1
            status_store.set_progress(completed, total)
            if "symbol" in result and "timeframe" in result:
                status_store.set_mode("optimize", result.get("symbol"), result.get("timeframe"))
            metrics = result.get("metrics")
            score = self._score(metrics)
            improved = False
            if score is not None:
                if best_score is None or score >= best_score + min_improvement:
                    best_score = score
                    best_result = result
                    if metrics:
                        status_store.set_metrics(metrics)
                    improved = True
            return improved

        status_store.set_mode("optimize", None, None)
        status_store.set_progress(0, total)
        try:
            if self._config.optimizer.enable_parallel and total > 1:
                workers = min(self._config.optimizer.max_workers or cpu_count(), cpu_count())
                with Pool(processes=workers, initializer=_set_optimizer_config, initargs=(self._config,)) as pool:
                    for item in pool.imap_unordered(_run_task, tasks):
                        handle_result(item)
            else:
                for task in tasks:
                    improved = handle_result(_run_task(task))
                    if improved:
                        no_improve = 0
                    else:
                        no_improve += 1
                    if allow_early_stop and no_improve >= (patience or 0):
                        logger.info(
                            "Early stopping optimizer after %s/%s evaluations (metric: %s)",
                            completed,
                            total,
                            score_label,
                        )
                        break
        finally:
            status_store.clear_progress()
            status_store.set_mode("idle", None, None)

        filtered = [result for result in results if "metrics" in result]
        filtered.sort(key=lambda item: self._score(item.get("metrics")) or float("-inf"), reverse=True)
        self._persist(filtered)
        return filtered

    def _markets(self) -> List[Tuple[str, str]]:
        if self._pairs is not None:
            allowed = set(self._limited_symbols()) if self._config.optimizer.max_symbols_per_run else None
            markets = [pair for pair in self._pairs if pair[1] in self.timeframes]
            if allowed is not None:
                markets = [pair for pair in markets if pair[0] in allowed]
            return markets
        return [(symbol, timeframe) for symbol in self._limited_symbols() for timeframe in self.timeframes]

    def _persist(self, results: List[Dict[str, Any]]) -> None:
        ensure_directories(self._config.paths, extra=[self._config.paths.optimization_dir])
        json_path = self._config.paths.optimization_dir / "best_params.json"
        csv_path = self._config.paths.optimization_dir / "best_params.csv"
        map_path = self._config.paths.optimization_dir / "best_params_by_market.json"

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

    def _score(self, metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        if not metrics:
            return None
        key = self._config.optimizer.score_metric or "total_pnl"
        value = metrics.get(key)
        if value is None and key != "total_pnl":
            value = metrics.get("total_pnl")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _best_map(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for record in results:
            symbol = record["symbol"]
            timeframe = record["timeframe"]
            metrics = record.get("metrics")
            score = self._score(metrics)
            if score is None:
                continue
            best_for_symbol = grouped.setdefault(symbol, {})
            current = best_for_symbol.get(timeframe)
            current_score = self._score(current.get("metrics") if current else None)
            if current is None or current_score is None or score > current_score:
                best_for_symbol[timeframe] = {"params": record["params"], "metrics": metrics or {}}
        return grouped


def load_best_parameters(cfg: BotConfig, symbol: str, timeframe: str) -> Optional[ParamDict]:
    """Load the best saved parameters for a market, if available."""
    candidates = [
        cfg.paths.optimization_dir / "learned_params.json",
        cfg.paths.optimization_dir / "hyper_params.json",
        cfg.paths.optimization_dir / "best_params_by_market.json",
        cfg.paths.optimization_dir / "best_params.json",
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
            return _normalize(params, cfg)
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
