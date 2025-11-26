"""CLI entrypoint for the Binance futures bot."""
from __future__ import annotations

import argparse
import asyncio
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import uvicorn

from dataclasses import asdict

from bot import data
from bot.backtester import Backtester
from bot.backtest_core_bridge import run_core_backtest
from bot.core.config import BotConfig, ensure_directories, load_config
from bot.exchange_info import ExchangeInfoManager
from bot.execution.client_factory import build_data_client, build_trading_client
from bot.learning import TradeLearningStore
from bot.optimizer import Optimizer
from bot.optimization import HyperparameterOptimizer
from bot.optimization_loader import load_best_params
from bot.portfolio import build_portfolio_meta, select_top_markets
from bot.pipeline import FullCycleRunner
from bot.simulator import DryRunner, MultiSymbolDryRunner
from bot.live_trader import LiveTrader
from bot.rl.agents import ActorCriticAgent
from bot.rl.env import FuturesTradingEnv
from bot.rl.policy_store import RLPolicyStore, _policy_name
from bot.rl.trainer import RLTrainer
from bot.rl.types import compute_baseline_hash
from bot.status import status_store
from bot.strategies import StrategyParameters, build_parameters
from bot.utils.logger import get_logger

logger = get_logger(__name__)


CommandHandler = Callable[[BotConfig, argparse.Namespace], None]


_RL_STORE: RLPolicyStore | None = None
_RL_STORE_KEY: Optional[str] = None


def _resolve_rl_store(cfg: BotConfig) -> RLPolicyStore | None:
    global _RL_STORE, _RL_STORE_KEY
    if not (cfg.runtime.use_rl_policy and cfg.rl.enabled):
        return None
    key = str(cfg.paths.optimization_dir)
    if _RL_STORE is not None and _RL_STORE_KEY == key:
        return _RL_STORE
    try:
        _RL_STORE = RLPolicyStore(cfg)
        _RL_STORE_KEY = key
    except Exception as exc:  # noqa: BLE001 - log and disable
        logger.warning("Unable to initialize RL policy store: %s", exc)
        _RL_STORE = None
        _RL_STORE_KEY = None
        return None
    return _RL_STORE


def _report_rl_state(
    *, active: bool, applied: bool, context: str, reason: Optional[str], symbol: Optional[str], interval: Optional[str]
) -> None:
    status_store.set_rl_state(
        {
            "active": bool(active),
            "applied": bool(applied),
            "context": context,
            "reason": reason,
            "symbol": symbol,
            "timeframe": interval,
        }
    )
def policy_guardrails(
    cfg: BotConfig,
    *,
    baseline_hash: str,
    policy_baseline_hash: str,
    param_deviation: float,
    metrics: Dict[str, float],
) -> tuple[bool, str]:
    if policy_baseline_hash != baseline_hash:
        return False, "baseline_mismatch"
    if param_deviation > cfg.rl.max_param_deviation_from_baseline:
        return False, f"deviation_{param_deviation:.3f}"
    val_mean = metrics.get("val_reward_mean")
    if val_mean is None:
        return False, "missing_validation_metrics"
    if val_mean < cfg.rl.min_validation_reward:
        return False, "val_reward_below_threshold"
    return True, ""


def load_rl_overrides(
    cfg: BotConfig,
    symbol: str,
    interval: str,
    *,
    rl_context: str,
    allow_rl: bool,
    ) -> Optional[Dict[str, float]]:
    active = cfg.runtime.use_rl_policy and cfg.rl.enabled
    if not active:
        _report_rl_state(
            active=False,
            applied=False,
            context=rl_context,
            reason="rl_disabled",
            symbol=symbol,
            interval=interval,
        )
        return None
    if not allow_rl:
        _report_rl_state(
            active=True,
            applied=False,
            context=rl_context,
            reason="context_blocked",
            symbol=symbol,
            interval=interval,
        )
        logger.info("RL overrides blocked for %s %s (%s context)", symbol, interval, rl_context)
        return None
    store = _resolve_rl_store(cfg)
    if not store:
        _report_rl_state(
            active=True,
            applied=False,
            context=rl_context,
            reason="store_unavailable",
            symbol=symbol,
            interval=interval,
        )
        logger.warning("RL policy store unavailable; skipping overrides for %s %s", symbol, interval)
        return None
    policy_name = _policy_name(symbol, interval)
    payload = store.load_latest_policy_params(policy_name)
    if not payload:
        _report_rl_state(
            active=True,
            applied=False,
            context=rl_context,
            reason="no_policy",
            symbol=symbol,
            interval=interval,
        )
        return None
    params, version, _run_metadata = payload
    baseline_hash = compute_baseline_hash(cfg.strategy.default_parameters)
    allowed, reason = policy_guardrails(
        cfg,
        baseline_hash=baseline_hash,
        policy_baseline_hash=version.baseline_params_hash,
        param_deviation=version.param_deviation,
        metrics=version.metrics,
    )
    if not allowed:
        _report_rl_state(
            active=True,
            applied=False,
            context=rl_context,
            reason=reason,
            symbol=symbol,
            interval=interval,
        )
        logger.warning(
            "RL overrides rejected for %s %s version=%s: %s",
            symbol,
            interval,
            version.version_id,
            reason,
        )
        return None
    _report_rl_state(
        active=True,
        applied=True,
        context=rl_context,
        reason=None,
        symbol=symbol,
        interval=interval,
    )
    logger.info(
        "Applying RL policy overrides for %s %s (context=%s) version=%s",
        symbol,
        interval,
        rl_context,
        version.version_id,
    )
    return params


def _resolve_device(cfg: BotConfig, preferred: Optional[str] = None) -> str:
    target = (preferred or cfg.rl.device_preference).lower()
    if target != "auto":
        return target
    try:
        import torch

        if torch.cuda.is_available():  # type: ignore[attr-defined]
            return "cuda"
    except Exception:  # noqa: BLE001 - fall back silently
        pass
    return "cpu"


def _resolve_download_symbols(cfg: BotConfig, symbols: Optional[Sequence[str]]) -> list[str]:
    if not symbols:
        return list(cfg.universe.default_symbols)
    tokens = [token.upper() for token in symbols if token]
    if not tokens:
        return list(cfg.universe.default_symbols)
    if len(tokens) == 1 and tokens[0] == "ALL":
        # Align downloads with the demo symbol universe to avoid fetching disabled markets.
        return list(cfg.universe.demo_symbols)
    return tokens


def cmd_download(cfg: BotConfig, args: argparse.Namespace) -> None:
    symbols = _resolve_download_symbols(cfg, args.symbols)
    intervals = args.intervals or cfg.universe.timeframes
    for symbol in symbols:
        for interval in intervals:
            try:
                data.download_klines(cfg, symbol, interval, limit=args.limit)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s %s download: %s", symbol, interval, exc)


def _resolve_strategy_overrides(
    cfg: BotConfig,
    symbol: str,
    interval: str,
    use_best: bool,
    *,
    rl_context: str,
) -> tuple[Optional[Dict[str, float]], str]:
    allow_rl = cfg.rl.enabled and cfg.runtime.use_rl_policy
    if rl_context == "live" and not cfg.rl.apply_to_live:
        allow_rl = False
    overrides: Optional[Dict[str, float]] = load_rl_overrides(
        cfg,
        symbol,
        interval,
        rl_context=rl_context,
        allow_rl=allow_rl,
    )
    if overrides is not None:
        return overrides, "rl_override"
    allow_best = use_best and cfg.runtime.use_optimizer_output
    if use_best and not cfg.runtime.use_optimizer_output:
        logger.info("Optimizer overrides disabled via BOT_USE_OPTIMIZER_OUTPUT=0")
    if allow_best:
        optimized = load_best_params(cfg, symbol, interval)
        if optimized:
            return optimized, "optimized"
        logger.warning("No optimized parameters found for %s %s; using defaults", symbol, interval)
    return None, "default"


def _build_strategy_params(
    cfg: BotConfig,
    symbol: str,
    interval: str,
    use_best: bool,
    *,
    rl_context: str,
) -> StrategyParameters:
    overrides, _ = _resolve_strategy_overrides(cfg, symbol, interval, use_best, rl_context=rl_context)
    return build_parameters(cfg, overrides)


def build_strategy_params_with_meta(
    cfg: BotConfig,
    symbol: str,
    interval: str,
    use_best: bool,
    *,
    rl_context: str,
) -> tuple[StrategyParameters, Optional[Dict[str, float]], str]:
    overrides, source = _resolve_strategy_overrides(cfg, symbol, interval, use_best, rl_context=rl_context)
    params = build_parameters(cfg, overrides)
    overrides_copy = dict(overrides) if overrides else None
    return params, overrides_copy, source


def cmd_backtest(cfg: BotConfig, args: argparse.Namespace) -> None:
    params, _override_payload, params_source = build_strategy_params_with_meta(
        cfg,
        args.symbol,
        args.interval,
        getattr(args, "use_best", False),
        rl_context="backtest",
    )
    params_dict = asdict(params)
    logger.info(
        "Using %s parameters for %s %s: %s",
        params_source,
        args.symbol,
        args.interval,
        params_dict,
    )
    candles = data.load_local_candles(cfg, args.symbol, args.interval)
    engine_choice = getattr(args, "engine", "core")

    if engine_choice == "core":
        result = run_core_backtest(
            cfg,
            args.symbol,
            args.interval,
            candles,
            params,
            params_dict,
            params_source,
        )
        trade_count = len(result.get("trades", []))
        logger.info(
            "Core backtest complete",
            extra={"symbol": args.symbol, "interval": args.interval, "trades": trade_count, "metrics": result.get("metrics", {})},
        )
    else:
        exchange = ExchangeInfoManager(cfg, client=build_data_client(cfg))
        exchange.refresh(force=True)
        backtester = Backtester(cfg, exchange)
        result = backtester.run(args.symbol, args.interval, candles, params)
        logger.info("Legacy backtest metrics: %s", result["metrics"])

def cmd_optimize(cfg: BotConfig, args: argparse.Namespace) -> None:
    symbols = args.symbols or list(cfg.universe.default_symbols)
    intervals = args.intervals or list(cfg.universe.timeframes)
    optimizer = Optimizer(symbols, intervals, cfg=cfg)
    optimizer.run()


def cmd_train_all(cfg: BotConfig, _: argparse.Namespace) -> None:
    optimizer = Optimizer(cfg.universe.default_symbols, cfg.universe.timeframes, cfg=cfg)
    optimizer.run()


def cmd_self_tune(cfg: BotConfig, args: argparse.Namespace) -> None:
    symbols = _resolve_download_symbols(cfg, args.symbols) if args.symbols else list(cfg.universe.default_symbols)
    intervals = args.intervals or list(cfg.universe.timeframes)
    rounds = args.rounds or 3
    top_k = args.top or 5
    hyper = HyperparameterOptimizer(cfg, symbols, intervals, rounds=rounds, top_k=top_k)
    results = hyper.run()
    if not results:
        logger.warning("Hyperparameter optimizer finished without results")
        return
    best = results[0]
    logger.info(
        "Best adaptive combo %s %s -> %s",
        best.get("symbol"),
        best.get("timeframe"),
        best.get("metrics"),
    )


def _demo_symbol_universe(cfg: BotConfig, exchange: ExchangeInfoManager) -> list[str]:
    client = exchange.client
    ordered_demo = list(cfg.universe.demo_symbols)
    if not client:
        logger.info("Demo exchange client unavailable; using configured demo symbols: %s", ", ".join(ordered_demo))
        return ordered_demo
    try:
        payload: Dict[str, Any] = client.exchange_info()  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to query demo exchangeInfo: %s", exc)
        return ordered_demo
    allowed = {symbol.upper() for symbol in ordered_demo}
    tradable: set[str] = set()
    for info in payload.get("symbols", []) or []:
        if info.get("contractType") != "PERPETUAL":
            continue
        if (info.get("quoteAsset") or "").upper() != "USDT":
            continue
        symbol = info.get("symbol")
        if not symbol:
            continue
        symbol_code = str(symbol).upper()
        if symbol_code not in allowed:
            continue
        status = (info.get("status") or "").upper() or "TRADING"
        if status in {"TRADING", "PENDING_TRADING"}:
            tradable.add(symbol_code)
    missing = [sym for sym in ordered_demo if sym not in tradable]
    if missing:
        logger.warning("Demo symbols unavailable or disabled: %s", ", ".join(missing))
    result = [sym for sym in ordered_demo if sym in tradable]
    if result:
        logger.info("Validated demo symbol universe: %s", ", ".join(result))
        return result
    logger.warning("No demo symbols validated via exchangeInfo; falling back to configured list")
    return ordered_demo


def _all_supported_symbols(cfg: BotConfig, exchange: ExchangeInfoManager, *, demo_mode: bool = False) -> list[str]:
    if demo_mode:
        universe = _demo_symbol_universe(cfg, exchange)
        if universe:
            logger.info("Demo mode symbol universe resolved to: %s", ", ".join(universe))
            return universe
        logger.warning("Falling back to configured demo symbols for demo-live session")
        fallback = list(cfg.universe.demo_symbols)
        logger.info("Demo mode fallback symbols: %s", ", ".join(fallback))
        return fallback
    symbols = sorted(exchange.symbols.keys())
    if symbols:
        return symbols
    logger.warning("Exchange info cache empty; falling back to default symbols")
    return list(cfg.universe.default_symbols)


def _resolve_symbols(
    cfg: BotConfig,
    symbol: Optional[str],
    symbols_arg: Optional[str],
    exchange: ExchangeInfoManager,
    *,
    demo_mode: bool = False,
) -> list[str]:
    if symbols_arg:
        if symbols_arg.strip().upper() == "ALL":
            return _all_supported_symbols(cfg, exchange, demo_mode=demo_mode)
        candidates = [token.strip().upper() for token in symbols_arg.split(",") if token.strip()]
    elif symbol:
        candidates = [symbol.upper()]
    else:
        raise ValueError("Provide --symbol or --symbols for dry-run")
    available = set(exchange.symbols.keys())
    if available:
        validated = [sym for sym in candidates if sym in available]
        missing = [sym for sym in candidates if sym not in available]
        if missing:
            logger.warning("Ignoring symbols missing from exchange info: %s", ", ".join(missing))
    else:
        validated = candidates
        missing = []
    if not validated:
        raise ValueError("No valid symbols resolved for dry-run")
    logger.info(
        "Resolved %s symbols: %s",
        "demo" if demo_mode else "standard",
        ", ".join(validated),
    )
    return validated


def _build_markets(
    cfg: BotConfig,
    symbols: Sequence[str],
    timeframe: str,
    use_best: bool,
    *,
    rl_context: str,
) -> list[tuple[str, str, StrategyParameters]]:
    markets: list[tuple[str, str, StrategyParameters]] = []
    for sym in symbols:
        params = _build_strategy_params(cfg, sym, timeframe, use_best, rl_context=rl_context)
        markets.append((sym, timeframe, params))
    return markets


def cmd_dry_run(cfg: BotConfig, args: argparse.Namespace) -> None:
    exchange = ExchangeInfoManager(cfg, client=build_data_client(cfg))
    exchange.refresh(force=True)
    use_best = getattr(args, "use_best", False)
    symbols = _resolve_symbols(cfg, args.symbol, getattr(args, "symbols", None), exchange)
    timeframe = args.interval
    if len(symbols) == 1:
        params = _build_strategy_params(cfg, symbols[0], timeframe, use_best, rl_context="dry_run")
        runner = DryRunner(symbols[0], timeframe, exchange, params, cfg)
    else:
        markets = _build_markets(cfg, symbols, timeframe, use_best, rl_context="dry_run")
        portfolio_meta = {
            "label": f"MANUAL ({len(symbols)})",
            "symbols": [{"symbol": sym, "timeframe": timeframe} for sym in symbols],
            "timeframe": timeframe,
            "metric": "manual",
        }
        runner = MultiSymbolDryRunner(markets, exchange, cfg, mode_label="dry_run_multi", portfolio_meta=portfolio_meta)
    asyncio.run(runner.run())


def cmd_dry_run_portfolio(cfg: BotConfig, args: argparse.Namespace) -> None:
    exchange = ExchangeInfoManager(cfg, client=build_data_client(cfg))
    exchange.refresh()
    timeframe = args.interval
    metric = args.metric or cfg.runtime.portfolio_metric
    top_n = args.top or cfg.runtime.top_symbols
    selection = select_top_markets(cfg, timeframe, top_n, metric)
    use_best = getattr(args, "use_best", False)
    if selection:
        markets: list[tuple[str, str, StrategyParameters]] = []
        for entry in selection:
            symbol = entry.get("symbol")
            if not symbol:
                continue
            raw_params = entry.get("params")
            params = (
                build_parameters(cfg, raw_params)
                if raw_params
                else _build_strategy_params(cfg, symbol, timeframe, use_best, rl_context="dry_run_portfolio")
            )
            markets.append((symbol, timeframe, params))
        portfolio_meta = build_portfolio_meta(selection, metric)
    else:
        logger.warning("No optimization results available; falling back to ALL symbols")
        symbols = _all_supported_symbols(cfg, exchange)
        markets = _build_markets(cfg, symbols, timeframe, use_best, rl_context="dry_run_portfolio")
        portfolio_meta = {
            "label": f"ALL ({len(symbols)})",
            "metric": metric,
            "symbols": [{"symbol": sym, "timeframe": timeframe} for sym in symbols],
            "timeframe": timeframe,
        }
    if not markets:
        raise RuntimeError("Portfolio dry-run requires at least one market")
    runner = MultiSymbolDryRunner(markets, exchange, cfg, mode_label="dry_run_portfolio", portfolio_meta=portfolio_meta)
    asyncio.run(runner.run())


def cmd_demo_live(cfg: BotConfig, args: argparse.Namespace) -> None:
    if not cfg.runtime.live_trading:
        raise RuntimeError(
            "Live trading is disabled. Set runtime.live_trading = True or export BOT_ENABLE_DEMO_LIVE=1 "
            "to acknowledge the risk before running."
        )
    if not cfg.runtime.use_testnet:
        raise RuntimeError(
            "Demo-live mode requires runtime.use_testnet=True so orders stay on the Futures Testnet."
            " Set BOT_USE_TESTNET=1 or adjust your config before retrying."
        )
    base_url = cfg.runtime.testnet_base_url.strip()
    if base_url.lower().endswith("fapi.binance.com") and "demo" not in base_url.lower() and "testnet" not in base_url.lower():
        raise RuntimeError(
            "Demo-live cannot target Binance mainnet endpoints. Configure BOT_TESTNET_BASE_URL"
            " with the demo-fapi or testnet URL."
        )
    logger.info(
        "RUN_MODE=demo-live (Binance Futures Testnet) | rest_base=%s",
        base_url,
    )
    trading_client = build_trading_client(cfg)
    exchange = ExchangeInfoManager(cfg, client=trading_client)
    exchange.refresh(force=True)
    use_best = getattr(args, "use_best", False)
    symbols = _resolve_symbols(cfg, args.symbol, getattr(args, "symbols", None), exchange, demo_mode=True)
    markets = _build_markets(cfg, symbols, args.interval, use_best, rl_context="live")
    logger.info("Initializing demo-live runner for symbols: %s", ", ".join(symbols))
    portfolio_meta = {
        "label": f"DEMO-LIVE ({len(symbols)})",
        "symbols": [{"symbol": sym, "timeframe": args.interval} for sym in symbols],
        "timeframe": args.interval,
        "metric": "demo_live",
    }
    learning_store = TradeLearningStore(cfg) if cfg.runtime.use_learning_store else None
    runner = LiveTrader(
        markets,
        exchange,
        cfg,
        client=trading_client,
        portfolio_meta=portfolio_meta,
        learning_store=learning_store,
        mode_label="demo-live",
    )
    asyncio.run(runner.run())


def cmd_train_rl(cfg: BotConfig, args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    interval = args.interval
    window = args.window or cfg.rl.lookback_window
    reward_scheme = args.reward or cfg.rl.reward_scheme
    max_steps = args.max_steps or cfg.rl.max_steps_per_episode
    episodes = args.episodes or cfg.rl.episodes
    checkpoint_interval = args.checkpoint_interval or cfg.rl.checkpoint_interval
    device = _resolve_device(cfg, getattr(args, "device", None))
    env = FuturesTradingEnv(symbol, interval, cfg, window=window, reward_scheme=reward_scheme, max_steps=max_steps)
    agent = ActorCriticAgent(env.observation_size, env.action_space, cfg, device=device)
    trainer = RLTrainer(env, agent, cfg)
    summary = trainer.train(episodes=episodes, checkpoint_interval=checkpoint_interval)
    logger.info(
        "RL training completed for %s %s | avg_reward=%.4f best_reward=%.4f",
        symbol,
        interval,
        summary.get("avg_reward", 0.0),
        summary.get("best_reward", 0.0),
    )


def cmd_api(cfg: BotConfig, args: argparse.Namespace) -> None:
    host = getattr(args, "host", cfg.runtime.api_host)
    port = int(getattr(args, "port", cfg.runtime.api_port))
    reload = bool(getattr(args, "reload", False))
    log_level = getattr(args, "log_level", cfg.runtime.log_level).lower()
    uvicorn.run("bot.api:app", host=host, port=port, reload=reload, log_level=log_level)


def cmd_full_cycle(cfg: BotConfig, _: argparse.Namespace) -> None:
    FullCycleRunner(cfg).run()


def build_parser(cfg: BotConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance USDC futures experimental bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download historical candles")
    download.add_argument("--symbols", nargs="*", help="Symbols such as BTCUSDT; defaults to configured list")
    download.add_argument("--intervals", nargs="*", help="Intervals like 1m 5m")
    download.add_argument("--limit", type=int, default=1500, help="Number of klines to request per batch")
    download.set_defaults(func=cmd_download)

    backtest = subparsers.add_parser("backtest", help="Run a single backtest on cached data")
    backtest.add_argument("--symbol", required=True, help="Symbol like BTCUSDT")
    backtest.add_argument("--interval", required=True, help="Interval such as 1m or 1h")
    backtest.add_argument(
        "--use-optimized",
        dest="use_best",
        action="store_true",
        help="Apply optimized parameters from optimization_results directory",
    )
    backtest.add_argument("--use-best", dest="use_best", action="store_true", help=argparse.SUPPRESS)
    backtest.add_argument(
        "--engine",
        choices=("core", "legacy"),
        default="core",
        help="Select the backtest implementation (default: core EMA pipeline)",
    )
    backtest.set_defaults(func=cmd_backtest)

    optimize = subparsers.add_parser("optimize", help="Launch parameter optimization")
    optimize.add_argument("--symbols", nargs="*", help="Symbols to optimize; default is configured universe")
    optimize.add_argument("--intervals", nargs="*", help="Intervals to optimize; default is configured timeframes")
    optimize.set_defaults(func=cmd_optimize)

    train_all = subparsers.add_parser("train-all", help="Optimize all configured symbols and intervals")
    train_all.set_defaults(func=cmd_train_all)

    self_tune = subparsers.add_parser("self-tune", help="Adaptive multi-round hyperparameter tuning")
    self_tune.add_argument("--symbols", nargs="*", help="Symbols to include in the search")
    self_tune.add_argument("--intervals", nargs="*", help="Intervals to include in the search")
    self_tune.add_argument("--rounds", type=int, default=3, help="Number of refinement rounds")
    self_tune.add_argument("--top", type=int, default=5, help="Number of top performers to keep each round")
    self_tune.set_defaults(func=cmd_self_tune)

    dry_run = subparsers.add_parser("dry-run", help="Start paper trading for one or more symbols")
    dry_run.add_argument("--symbol", help="Single symbol such as BTCUSDT")
    dry_run.add_argument(
        "--symbols",
        help="Comma-separated list (e.g. BTCUSDT,ETHUSDT) or ALL to include every supported symbol",
    )
    dry_run.add_argument("--interval", required=True, help="Interval to trade")
    dry_run.add_argument("--use-best", action="store_true", help="Apply optimized parameters when available")
    dry_run.set_defaults(func=cmd_dry_run)

    dry_run_portfolio = subparsers.add_parser(
        "dry-run-portfolio",
        help="Start paper trading on the top-ranked portfolio",
    )
    dry_run_portfolio.add_argument("--interval", required=True, help="Interval to trade")
    dry_run_portfolio.add_argument("--metric", help="Portfolio ranking metric override", default=None)
    dry_run_portfolio.add_argument("--top", type=int, help="Override the number of symbols to include", default=None)
    dry_run_portfolio.add_argument(
        "--use-best",
        action="store_true",
        help="Fallback to best saved parameters if optimizer output is missing",
    )
    dry_run_portfolio.set_defaults(func=cmd_dry_run_portfolio)

    demo_live = subparsers.add_parser("demo-live", help="Execute demo trades on the Binance futures testnet")
    demo_live.add_argument("--symbol", help="Single symbol such as BTCUSDT")
    demo_live.add_argument(
        "--symbols",
        help="Comma-separated list (e.g. BTCUSDT,ETHUSDT) or ALL to include every supported symbol",
    )
    demo_live.add_argument("--interval", required=True, help="Interval to trade")
    demo_live.add_argument("--use-best", action="store_true", help="Apply optimized parameters when available")
    demo_live.set_defaults(func=cmd_demo_live)

    train_rl = subparsers.add_parser("train-rl", help="Train the reinforcement learning agent")
    train_rl.add_argument("--symbol", required=True, help="Symbol like BTCUSDT")
    train_rl.add_argument("--interval", required=True, help="Interval such as 1m or 5m")
    train_rl.add_argument("--window", type=int, help="Override lookback window", default=None)
    train_rl.add_argument("--reward", help="Reward scheme name", default=None)
    train_rl.add_argument("--max-steps", type=int, help="Max steps per episode", default=None)
    train_rl.add_argument("--episodes", type=int, help="Episode count override", default=None)
    train_rl.add_argument("--checkpoint-interval", type=int, help="Save checkpoints every N episodes", default=None)
    train_rl.add_argument("--device", help="Device preference: cpu/cuda/auto", default=None)
    train_rl.set_defaults(func=cmd_train_rl)

    api_cmd = subparsers.add_parser("api", help="Start the FastAPI dashboard server")
    api_cmd.add_argument("--host", default=cfg.runtime.api_host, help="Listening host")
    api_cmd.add_argument("--port", type=int, default=cfg.runtime.api_port, help="Listening port")
    api_cmd.add_argument("--reload", action="store_true", help="Enable autoreload (dev only)")
    api_cmd.add_argument("--log-level", default=cfg.runtime.log_level, help="Uvicorn log level")
    api_cmd.set_defaults(func=cmd_api)

    full_cycle = subparsers.add_parser("full-cycle", help="Run the download/backtest/optimize + dry-run pipeline")
    full_cycle.set_defaults(func=cmd_full_cycle)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = load_config()
    ensure_directories(cfg.paths)
    status_store.configure(log_dir=cfg.paths.log_dir, default_balance=cfg.runtime.paper_account_balance)
    parser = build_parser(cfg)
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        raise SystemExit(1)
    func(cfg, args)


if __name__ == "__main__":
    main()
