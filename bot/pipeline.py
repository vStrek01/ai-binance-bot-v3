"""Automation pipeline for the full download→backtest→optimize→dry-run cycle."""
from __future__ import annotations

import asyncio
import socket
from typing import List, Sequence, Tuple

import uvicorn
from bot import data
from bot.backtester import Backtester
from bot.core.config import BotConfig, ensure_directories
from bot.execution.client_factory import build_data_client
from bot.exchange_info import ExchangeInfoManager
from bot.optimizer import Optimizer, load_best_parameters
from bot.simulator import MultiSymbolDryRunner
from bot.status import status_store
from bot.strategies import StrategyParameters, build_parameters
from bot.strategy_mode import resolve_strategy_mode
from bot.utils.logger import get_logger

logger = get_logger(__name__)


class FullCycleRunner:
    """Coordinates downloads, validation, backtests, optimization, and dry-run."""

    def __init__(
        self,
        cfg: BotConfig,
        symbols: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
    ) -> None:
        self._config = cfg
        self.exchange = ExchangeInfoManager(cfg, client=build_data_client(cfg))
        if not self.exchange.symbols:
            logger.info("Exchange info cache empty; refreshing metadata")
            self.exchange.refresh(force=True)
        self.symbols = list(symbols) if symbols else sorted(self.exchange.symbols.keys())
        if not self.symbols:
            raise RuntimeError("No USDC symbols available in exchange info cache")
        self.timeframes = list(timeframes or cfg.universe.timeframes)
        if not self.timeframes:
            raise RuntimeError("No timeframes configured")
        self.market_pairs: List[Tuple[str, str]] = [(sym, tf) for sym in self.symbols for tf in self.timeframes]
        self.ready_markets: List[Tuple[str, str]] = []

    def run(self) -> None:
        cfg = self._config
        logger.info("Starting full learning cycle for %s symbols x %s timeframes", len(self.symbols), len(self.timeframes))
        ensure_directories(cfg.paths)
        status_store.update_balance(cfg.runtime.paper_account_balance)
        status_store.set_open_pnl(0.0)
        status_store.set_positions([])
        try:
            self._download_all()
            self._validate_all()
            self._warmup_backtests()
            self._optimize()
            markets = self._build_market_params()
            asyncio.run(self._launch_runtime(markets))
        except KeyboardInterrupt:  # pragma: no cover - manual stop
            logger.info("Full cycle interrupted by user")
            status_store.set_mode("idle", None, None)

    def _download_all(self) -> None:
        total = len(self.market_pairs)
        status_store.set_mode("download", None, None)
        status_store.set_progress(0, total)
        for idx, (symbol, timeframe) in enumerate(self.market_pairs, start=1):
            try:
                data.ensure_local_candles(self._config, symbol, timeframe, min_rows=self._config.runtime.lookback_limit)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Download failed for %s %s: %s", symbol, timeframe, exc)
            finally:
                status_store.set_progress(idx, total)
        status_store.clear_progress()

    def _validate_all(self) -> None:
        total = len(self.market_pairs)
        status_store.set_mode("validate", None, None)
        status_store.set_progress(0, total)
        for idx, (symbol, timeframe) in enumerate(self.market_pairs, start=1):
            try:
                frame = data.load_local_candles(self._config, symbol, timeframe)
                data.validate_candles(frame, symbol, timeframe)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Validation failed for %s %s: %s", symbol, timeframe, exc)
                try:
                    refreshed = data.download_klines(
                        self._config,
                        symbol,
                        timeframe,
                        limit=self._config.runtime.lookback_limit,
                    )
                    data.validate_candles(refreshed, symbol, timeframe)
                except Exception as retry_exc:  # noqa: BLE001
                    logger.error("Unable to repair dataset %s %s: %s", symbol, timeframe, retry_exc)
            finally:
                status_store.set_progress(idx, total)
        status_store.clear_progress()

    def _warmup_backtests(self) -> None:
        defaults = build_parameters(self._config)
        total = len(self.market_pairs)
        status_store.set_mode("backtest", None, None)
        status_store.set_progress(0, total)
        backtester = Backtester(self._config, self.exchange)
        for idx, (symbol, timeframe) in enumerate(self.market_pairs, start=1):
            try:
                candles = data.load_local_candles(self._config, symbol, timeframe)
            except FileNotFoundError:
                logger.warning("Skipping baseline backtest; candles missing for %s %s", symbol, timeframe)
                status_store.set_progress(idx, total)
                continue
            try:
                result = backtester.run(symbol, timeframe, candles, defaults)
            except Exception as exc:  # noqa: BLE001
                logger.error("Backtest failed for %s %s: %s", symbol, timeframe, exc)
                status_store.set_progress(idx, total)
                continue
            metrics = result["metrics"]
            status_store.set_metrics(metrics)
            logger.info(
                "Baseline %s %s → trades=%s pnl=%.2f",
                symbol,
                timeframe,
                metrics.get("trades", 0),
                metrics.get("total_pnl", 0.0),
            )
            self.ready_markets.append((symbol, timeframe))
            status_store.set_progress(idx, total)
        status_store.clear_progress()
        if not self.ready_markets:
            raise RuntimeError("No markets produced successful baseline backtests")

    def _optimize(self) -> None:
        if not self.ready_markets:
            logger.warning("Skipping optimization because no markets are ready")
            return
        symbols = sorted({sym for sym, _ in self.ready_markets})
        timeframes = sorted({tf for _, tf in self.ready_markets})
        optimizer = Optimizer(symbols, timeframes, pairs=self.ready_markets, cfg=self._config)
        optimizer.run()

    def _build_market_params(self) -> List[Tuple[str, str, StrategyParameters]]:
        markets = self.ready_markets or self.market_pairs
        plans: List[Tuple[str, str, StrategyParameters]] = []
        for symbol, timeframe in markets:
            overrides = load_best_parameters(self._config, symbol, timeframe)
            if overrides:
                logger.info("Using optimized parameters for %s %s", symbol, timeframe)
            else:
                logger.info("Falling back to default parameters for %s %s", symbol, timeframe)
            plans.append((symbol, timeframe, build_parameters(self._config, symbol=symbol, overrides=overrides)))
        return plans

    async def _launch_runtime(self, markets: Sequence[Tuple[str, str, StrategyParameters]]) -> None:
        runner = MultiSymbolDryRunner(
            markets,
            self.exchange,
            self._config,
            strategy_mode=resolve_strategy_mode(self._config),
        )
        await asyncio.gather(runner.run(), self._serve_dashboard())

    async def _serve_dashboard(self) -> None:
        ensure_directories(self._config.paths)
        host = self._config.runtime.api_host
        port = self._config.runtime.api_port
        if not _port_available(host, port):
            logger.warning("Dashboard server skipped; %s:%s already in use", host, port)
            return
        config_kwargs = {
            "app": "bot.api:app",
            "host": host,
            "port": port,
            "log_level": "info",
        }
        server = uvicorn.Server(uvicorn.Config(**config_kwargs))
        try:
            await server.serve()
        except SystemExit as exc:
            if getattr(exc, "code", 0) not in (0, None):
                logger.warning("Dashboard server exited early (code=%s)", exc.code)


def _port_available(host: str, port: int) -> bool:
    """Return True when the TCP port is free to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
        return True


__all__ = ["FullCycleRunner"]
