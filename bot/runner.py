"""CLI entrypoint for the Binance futures bot."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import uvicorn

from bot import data
from bot.backtester import Backtester
from bot.backtest_core_bridge import run_core_backtest
from bot.core.config import BotConfig, ensure_directories, load_config
from bot.core.secrets import get_binance_api_key, get_binance_api_secret
from bot.crash_recovery import bootstrap_crash_recovery
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
from bot.risk.spec import build_core_risk_config
from bot.status import status_store
from bot.strategies import StrategyParameters, build_parameters
from bot.utils.logger import get_logger
from core.health import run_pre_trade_checks
from core.models import RiskConfig as CoreRiskConfig
from core.risk import RiskManager
from exchange.binance_client import BinanceClient
from exchange.symbols import SymbolResolver
from infra.alerts import send_alert
from infra.state_store import StateStore
import infra.logging as logging_utils

logger = get_logger(__name__)


CommandHandler = Callable[[BotConfig, argparse.Namespace], None]


_RL_STORE: RLPolicyStore | None = None
_RL_STORE_KEY: Optional[str] = None


@dataclass(slots=True)
class RuntimeContext:
    run_id: str
    state_store: StateStore
    symbol_resolver: SymbolResolver
    health_summary: Dict[str, Any]
    exchange_client: BinanceClient
    risk_manager: RiskManager


_RUNTIME_CONTEXT: RuntimeContext | None = None

def _enforce_run_mode(cfg: BotConfig) -> None:
    run_mode = cfg.run_mode
    use_testnet = cfg.exchange.use_testnet
    logging_utils.log_event(
        "runtime_mode_resolved",
        run_mode=run_mode,
        use_testnet=use_testnet,
        rest_base=cfg.exchange.rest_base_url,
        ws_market=cfg.exchange.ws_market_url,
        source="cli",
    )
    if run_mode == "demo-live" and not use_testnet:
        logging_utils.log_event(
            "runtime_mode_blocked",
            run_mode=run_mode,
            reason="demo_requires_testnet",
            rest_base=cfg.exchange.rest_base_url,
        )
        logger.critical("demo-live mode requires exchange.use_testnet=True")
        raise SystemExit(2)
    if run_mode == "live":
        env_flag = (os.getenv("BOT_LIVE_ENABLE") or "").strip().lower()
        if env_flag not in {"1", "true", "yes", "on"}:
            logging_utils.log_event("runtime_mode_blocked", run_mode="live", reason="env_guard")
            logger.critical("BOT_LIVE_ENABLE=1 required to start live trading")
            raise SystemExit(3)
        if use_testnet:
            logging_utils.log_event("runtime_mode_blocked", run_mode="live", reason="testnet_not_allowed")
            logger.critical("Live mode must point at mainnet endpoints")
            raise SystemExit(4)


def _resolve_run_id() -> str:
    env_id = os.getenv("BOT_RUN_ID")
    if env_id:
        return env_id.strip()
    return datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")


def _build_state_store(cfg: BotConfig, run_id: str) -> StateStore:
    state_dir = cfg.paths.data_dir / "state"
    return StateStore(run_id, base_dir=state_dir)


def _build_binance_client(cfg: BotConfig) -> BinanceClient:
    prefer_testnet = cfg.exchange.use_testnet
    api_key = get_binance_api_key(prefer_testnet=prefer_testnet, required=False)
    api_secret = get_binance_api_secret(prefer_testnet=prefer_testnet, required=False)
    config_key = (cfg.exchange.api_key or "").strip()
    config_secret = (cfg.exchange.api_secret or "").strip()
    effective_api_key = api_key or config_key
    effective_api_secret = api_secret or config_secret
    base_url = (cfg.exchange.rest_base_url or "").strip() or None
    _log_credential_source(
        cfg_source=bool(config_key or config_secret),
        env_source=bool(api_key or api_secret),
        effective_api_key=effective_api_key,
    )
    return BinanceClient(
        api_key=effective_api_key or "",
        api_secret=effective_api_secret or "",
        base_url=base_url,
        testnet=prefer_testnet,
    )


def _hash_api_key(value: str | None) -> str | None:
    if not value:
        return None
    digest = hashlib.sha256(value.encode()).hexdigest()
    return digest[:8]


def _log_credential_source(*, cfg_source: bool, env_source: bool, effective_api_key: str | None) -> None:
    if not (cfg_source or env_source or effective_api_key):
        logging_utils.log_event("EXCHANGE_CREDENTIAL_SOURCE", source="none", api_key_hash=None)
        return
    if cfg_source and env_source:
        source = "env+config"
    elif env_source:
        source = "env"
    else:
        source = "config"
    logging_utils.log_event(
        "EXCHANGE_CREDENTIAL_SOURCE",
        source=source,
        api_key_hash=_hash_api_key(effective_api_key),
    )


def _should_bootstrap_runtime(cfg: BotConfig) -> bool:
    return cfg.run_mode in {"dry-run", "demo-live", "live"}


def _maybe_bootstrap_runtime(cfg: BotConfig) -> RuntimeContext | None:
    global _RUNTIME_CONTEXT
    if not _should_bootstrap_runtime(cfg):
        return None
    if _RUNTIME_CONTEXT is not None:
        return _RUNTIME_CONTEXT

    run_id = _resolve_run_id()
    state_store = _build_state_store(cfg, run_id)
    exchange_client = _build_binance_client(cfg)
    symbol_resolver = SymbolResolver(exchange_client, symbols=cfg.symbols)
    account_provider: BinanceClient | None = None
    if cfg.run_mode in {"demo-live", "live"} and not cfg.runtime.dry_run:
        account_provider = exchange_client

    health_ok, health_meta = run_pre_trade_checks(
        cfg,
        exchange_client,
        symbol_resolver,
        run_id=run_id,
        account_provider=account_provider,
    )
    if not health_ok:
        send_alert(
            "PRE_TRADE_HEALTH_FAILED",
            severity="critical",
            message="Pre-trade health checks failed",
            run_mode=cfg.run_mode,
            checks=list(health_meta.values()),
        )
        logging_utils.log_event(
            "runtime_mode_blocked",
            run_mode=cfg.run_mode,
            reason="health_check_failed",
            checks=list(health_meta.values()),
            run_id=run_id,
        )
        raise SystemExit("Pre-trade health checks failed; see logs for details.")

    risk_config = build_core_risk_config(cfg)
    risk_manager = RiskManager(
        risk_config,
        symbol_resolver=symbol_resolver,
        state_store=state_store,
        run_id=run_id,
    )

    _RUNTIME_CONTEXT = RuntimeContext(
        run_id=run_id,
        state_store=state_store,
        symbol_resolver=symbol_resolver,
        health_summary=health_meta,
        exchange_client=exchange_client,
        risk_manager=risk_manager,
    )
    logging_utils.bind_log_context(run_id=run_id)
    logging_utils.log_event(
        "runtime_bootstrap_complete",
        run_mode=cfg.run_mode,
        run_id=run_id,
        checks=list(health_meta.values()),
    )
    return _RUNTIME_CONTEXT


def get_runtime_context() -> RuntimeContext | None:
    return _RUNTIME_CONTEXT


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


def _emit_demo_live_snapshot(cfg: BotConfig, *, symbols: Sequence[str], interval: str, dry_run: bool) -> None:
    equity = cfg.runtime.paper_account_balance
    logging_utils.log_event(
        "EQUITY_SNAPSHOT",
        run_mode=cfg.run_mode,
        mode="demo-live",
        dry_run=dry_run,
        use_testnet=cfg.exchange.use_testnet,
        rest_base=cfg.exchange.rest_base_url,
        ws_market=cfg.exchange.ws_market_url,
        ws_user=cfg.exchange.ws_user_url,
        leverage=cfg.risk.leverage,
        symbols=list(symbols),
        interval=interval,
        equity=equity,
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
    return build_parameters(cfg, symbol=symbol, overrides=overrides)


def build_strategy_params_with_meta(
    cfg: BotConfig,
    symbol: str,
    interval: str,
    use_best: bool,
    *,
    rl_context: str,
) -> tuple[StrategyParameters, Optional[Dict[str, float]], str]:
    overrides, source = _resolve_strategy_overrides(cfg, symbol, interval, use_best, rl_context=rl_context)
    params = build_parameters(cfg, symbol=symbol, overrides=overrides)
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
                build_parameters(cfg, symbol=symbol, overrides=raw_params)
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
    base_url = (cfg.exchange.rest_base_url or "").strip()
    if not base_url:
        raise RuntimeError("exchange.rest_base_url must be configured before running demo-live")
    lowered = base_url.lower()
    if lowered.endswith("fapi.binance.com") and "demo" not in lowered and "testnet" not in lowered:
        raise RuntimeError(
            "Demo-live cannot target Binance mainnet endpoints. Set BINANCE_REST_URL or exchange.rest_base_url to the demo-fapi/testnet host."
        )
    ws_market = (cfg.exchange.ws_market_url or "").strip()
    ws_user = (cfg.exchange.ws_user_url or "").strip() or "n/a"
    logger.info(
        "RUN_MODE=demo-live (Binance Futures Testnet) | rest_base=%s | ws_market=%s | ws_user=%s",
        base_url,
        ws_market,
        ws_user,
    )
    rl_live_enabled = cfg.runtime.use_rl_policy and cfg.rl.enabled and cfg.rl.apply_to_live
    if not rl_live_enabled:
        logger.info("RL overrides disabled for demo-live (set runtime.use_rl_policy=1 and rl.apply_to_live=1 to enable)")
    elif cfg.runtime.use_rl_policy:
        logger.warning("RL overrides enabled for demo-live; ensure guardrails are configured")
    if cfg.llm.enabled:
        logger.warning("LLM insights enabled for demo-live session. Disable via llm.enabled=false to keep deterministic baseline.")
    else:
        logger.info("LLM insights remain disabled for demo-live")
    runtime_ctx = get_runtime_context()
    state_store = runtime_ctx.state_store if runtime_ctx else None
    resume_run = getattr(args, "resume_run", None) or os.getenv("BOT_RESUME_RUN")
    use_best = getattr(args, "use_best", False)
    if cfg.runtime.dry_run:
        logger.info("BOT_DRY_RUN detected; running demo-live smoke test with paper execution only")
        data_client = build_data_client(cfg)
        exchange = ExchangeInfoManager(cfg, client=data_client)
        exchange.refresh(force=True)
        symbols = _resolve_symbols(cfg, args.symbol, getattr(args, "symbols", None), exchange, demo_mode=True)
        markets = _build_markets(cfg, symbols, args.interval, use_best, rl_context="live")
        _emit_demo_live_snapshot(cfg, symbols=symbols, interval=args.interval, dry_run=True)
        logger.info("Initializing demo-live dry-run runner for symbols: %s", ", ".join(symbols))
        portfolio_meta = {
            "label": f"DEMO-LIVE DRY ({len(symbols)})",
            "symbols": [{"symbol": sym, "timeframe": args.interval} for sym in symbols],
            "timeframe": args.interval,
            "metric": "demo_live_smoke",
        }
        runner = MultiSymbolDryRunner(
            markets,
            exchange,
            cfg,
            mode_label="demo-live-smoke",
            portfolio_meta=portfolio_meta,
        )
        asyncio.run(runner.run_cycles(1))
        return

    trading_client = build_trading_client(cfg)
    exchange = ExchangeInfoManager(cfg, client=trading_client)
    exchange.refresh(force=True)
    if resume_run:
        bootstrap_crash_recovery(
            base_dir=cfg.paths.data_dir / "state",
            resume_run_id=resume_run,
            trading_client=trading_client,
            state_store=state_store,
        )
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
        state_store=state_store,
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
    demo_live.add_argument(
        "--resume-run",
        help="Optional previous run_id to compare against for crash recovery",
    )
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
    _enforce_run_mode(cfg)
    ensure_directories(cfg.paths)
    runtime_ctx = _maybe_bootstrap_runtime(cfg)
    status_store.configure(log_dir=cfg.paths.log_dir, default_balance=cfg.runtime.paper_account_balance)
    status_store.set_runtime_context(
        run_mode=cfg.run_mode,
        use_testnet=cfg.runtime.use_testnet,
        rest_base_url=cfg.exchange.rest_base_url,
        ws_market_url=cfg.exchange.ws_market_url,
        ws_user_url=cfg.exchange.ws_user_url,
    )
    logging_utils.bind_log_context(
        run_mode=cfg.run_mode,
        use_testnet=cfg.runtime.use_testnet,
        rest_base_url=cfg.exchange.rest_base_url,
        ws_market_url=cfg.exchange.ws_market_url,
        ws_user_url=cfg.exchange.ws_user_url,
        run_id=runtime_ctx.run_id if runtime_ctx else None,
    )
    parser = build_parser(cfg)
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        raise SystemExit(1)
    func(cfg, args)


if __name__ == "__main__":
    main()
