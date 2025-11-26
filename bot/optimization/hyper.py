"""Adaptive hyperparameter search that refines ranges between rounds."""
from __future__ import annotations

import json
import statistics
from typing import Any, Dict, Iterable, List

from bot.core import config
from bot.optimizer import Optimizer
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterOptimizer:
    def __init__(
        self,
        symbols: Iterable[str],
        timeframes: Iterable[str],
        rounds: int = 3,
        top_k: int = 5,
    ) -> None:
        self.symbols = [sym.upper() for sym in symbols]
        self.timeframes = list(timeframes)
        self.rounds = max(1, rounds)
        self.top_k = max(1, top_k)
        self._baseline_space = {key: list(values) for key, values in config.strategy.parameter_space.items()}
        self._original_space = {key: list(values) for key, values in config.strategy.parameter_space.items()}

    def run(self) -> List[Dict[str, Any]]:
        best_results: List[Dict[str, Any]] = []
        current_space = {key: list(values) for key, values in self._baseline_space.items()}
        for round_index in range(1, self.rounds + 1):
            logger.info(
                "Hyperparameter round %s/%s with %s symbols x %s timeframes",
                round_index,
                self.rounds,
                len(self.symbols),
                len(self.timeframes),
            )
            self._apply_space(current_space)
            optimizer = Optimizer(self.symbols, self.timeframes)
            results = optimizer.run()
            if not results:
                logger.warning("Adaptive round %s produced no results; stopping early", round_index)
                break
            best_results = results[: self.top_k]
            current_space = self._refine_space(best_results)
        self._restore_space()
        if best_results:
            self._persist(best_results)
        return best_results

    def _apply_space(self, space: Dict[str, List[float]]) -> None:
        config.strategy.parameter_space = {key: list(values) for key, values in space.items()}

    def _restore_space(self) -> None:
        config.strategy.parameter_space = {key: list(values) for key, values in self._original_space.items()}

    def _refine_space(self, winners: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        refined: Dict[str, List[float]] = {}
        for key, baseline in self._baseline_space.items():
            values = [entry.get("params", {}).get(key) for entry in winners if isinstance(entry.get("params"), dict)]
            typed_values = [float(value) for value in values if isinstance(value, (int, float))]
            if not typed_values:
                refined[key] = list(baseline)
                continue
            default = config.strategy.default_parameters.get(key, typed_values[0])
            refined[key] = self._expand_candidates(key, typed_values, baseline, default)
        return refined

    def _expand_candidates(
        self,
        key: str,
        samples: List[float],
        baseline: List[float],
        default: float,
    ) -> List[float]:
        is_int = isinstance(default, int)
        center = statistics.mean(samples)
        spread = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        if spread == 0:
            spread = max(abs(center) * 0.1, 1.0 if is_int else 0.1)
        offsets = (-spread, 0.0, spread)
        min_allowed = min(baseline) if baseline else None
        max_allowed = max(baseline) if baseline else None
        candidates: List[float] = []
        for offset in offsets:
            candidate = center + offset
            if min_allowed is not None:
                candidate = max(min_allowed, candidate)
            if max_allowed is not None:
                candidate = min(max_allowed, candidate)
            candidates.append(candidate)
        candidates.extend(samples)
        deduped = sorted({int(round(value)) if is_int else round(value, 4) for value in candidates})
        if len(deduped) < 3 and baseline:
            deduped.extend(baseline[: 3 - len(deduped)])
        unique_values = sorted({float(value) for value in deduped})
        if is_int:
            return [float(int(round(value))) for value in unique_values]
        return unique_values

    def _persist(self, results: List[Dict[str, Any]]) -> None:
        config.ensure_directories([config.OPTIMIZATION_DIR])
        path = config.OPTIMIZATION_DIR / "hyper_params.json"
        payload = json.dumps(results, indent=2)
        atomic_write_text(path, payload)
        logger.info("Hyperparameter optimizer results saved to %s", path)


__all__ = ["HyperparameterOptimizer"]
