"""Adaptive hyperparameter search that refines ranges between rounds."""
from __future__ import annotations

import json
import statistics
from typing import Any, Dict, Iterable, List

from bot.core.config import BotConfig, ensure_directories
from bot.optimizer import Optimizer
from bot.utils.fileio import atomic_write_text
from bot.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterOptimizer:
    def __init__(
        self,
        cfg: BotConfig,
        symbols: Iterable[str],
        timeframes: Iterable[str],
        rounds: int = 3,
        top_k: int = 5,
    ) -> None:
        self._config = cfg
        self.symbols = [sym.upper() for sym in symbols]
        self.timeframes = list(timeframes)
        self.rounds = max(1, rounds)
        self.top_k = max(1, top_k)
        self._baseline_space = {key: list(values) for key, values in cfg.strategy.parameter_space.items()}
        self._default_params = dict(cfg.strategy.default_parameters)

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
            round_cfg = self._with_space(current_space)
            optimizer = Optimizer(self.symbols, self.timeframes, cfg=round_cfg)
            results = optimizer.run()
            if not results:
                logger.warning("Adaptive round %s produced no results; stopping early", round_index)
                break
            best_results = results[: self.top_k]
            current_space = self._refine_space(best_results)
        if best_results:
            self._persist(best_results)
        return best_results

    def _with_space(self, space: Dict[str, List[float]]) -> BotConfig:
        strategy = self._config.strategy.model_copy(
            update={"parameter_space": {key: list(values) for key, values in space.items()}}
        )
        return self._config.model_copy(update={"strategy": strategy})

    def _refine_space(self, winners: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        refined: Dict[str, List[float]] = {}
        for key, baseline in self._baseline_space.items():
            values = [entry.get("params", {}).get(key) for entry in winners if isinstance(entry.get("params"), dict)]
            typed_values = [float(value) for value in values if isinstance(value, (int, float))]
            if not typed_values:
                refined[key] = list(baseline)
                continue
            default = self._default_params.get(key, typed_values[0])
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
        ensure_directories(self._config.paths, extra=[self._config.paths.optimization_dir])
        path = self._config.paths.optimization_dir / "hyper_params.json"
        payload = json.dumps(results, indent=2)
        atomic_write_text(path, payload)
        logger.info("Hyperparameter optimizer results saved to %s", path)


__all__ = ["HyperparameterOptimizer"]
