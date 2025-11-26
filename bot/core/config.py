"""Dependency-injected configuration primitives for the trading bot."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple


class ConfigValidationError(ValueError):
    """Raised when configuration values fail validation."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as err:  # pragma: no cover - configuration guard
        raise ConfigValidationError(f"Environment variable {name} must be an integer") from err


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as err:  # pragma: no cover - configuration guard
        raise ConfigValidationError(f"Environment variable {name} must be numeric") from err


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None else default


def _env_tuple(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    tokens = [token.strip().upper() for token in raw.split(",") if token.strip()]
    return tuple(tokens) if tokens else default


@dataclass(frozen=True)
class PathConfig:
    base_dir: Path
    data_dir: Path
    results_dir: Path
    optimization_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class UniverseConfig:
    timeframes: Tuple[str, ...] = ("1m", "5m", "15m", "1h")
    demo_symbols: Tuple[str, ...] = ("BTCUSDT", "BCHUSDT", "ETHUSDT", "ETCUSDT", "LTCUSDT", "XRPUSDT")
    default_symbols: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "LTCUSDT")


@dataclass(frozen=True)
class RiskConfig:
    per_trade_risk: float = 0.005
    leverage: float = 3.0
    taker_fee: float = 0.0004
    maker_fee: float = 0.0002
    min_free_margin: float = 50.0
    margin_buffer: float = 0.1
    margin_warning_cooldown: float = 90.0
    margin_relief_factor: float = 1.15
    max_symbol_exposure: float = 0.1
    max_account_exposure: float = 0.25
    max_concurrent_symbols: int = 3
    max_daily_loss_pct: float = 0.02
    max_daily_loss_abs: float | None = None
    stop_trading_on_daily_loss: bool = True
    close_positions_on_daily_loss: bool = False
    daily_loss_lookback_hours: int = 24


@dataclass(frozen=True)
class BacktestConfig:
    initial_balance: float = 1_000.0
    fee_model: str = "taker"
    slippage_bps: float = 1.0
    max_bars: int = 50_000
    realism_level: Literal["toy", "standard", "aggressive"] = "standard"
    enable_funding_costs: bool = False
    funding_rate_bps: float = 1.0
    funding_interval_hours: int = 8
    enable_latency: bool = False
    latency_ms: int = 250


@dataclass(frozen=True)
class StrategyConfig:
    default_parameters: Dict[str, float] = field(
        default_factory=lambda: {
            "fast_ema": 13,
            "slow_ema": 34,
            "rsi_length": 14,
            "rsi_overbought": 60,
            "rsi_oversold": 40,
            "atr_period": 14,
            "atr_stop": 1.6,
            "atr_target": 2.2,
            "cooldown_bars": 2,
            "hold_bars": 90,
        }
    )
    parameter_space: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "fast_ema": [8, 13, 21],
            "slow_ema": [34, 55, 89],
            "rsi_overbought": [58, 60, 65],
            "rsi_oversold": [35, 40, 45],
            "atr_stop": [1.4, 1.6, 1.9],
            "atr_target": [1.8, 2.2, 2.8],
        }
    )


@dataclass(frozen=True)
class OptimizerConfig:
    enable_parallel: bool = True
    max_workers: int = 4
    max_param_combinations: int | None = None
    max_symbols_per_run: int | None = None
    randomize: bool = False
    search_mode: Literal["grid", "random"] = "grid"
    random_subset: int | None = None
    random_seed: int | None = None
    score_metric: str = "total_pnl"
    early_stop_patience: int | None = None
    min_improvement: float = 0.0


@dataclass(frozen=True)
class RuntimeConfig:
    dry_run: bool = True
    use_testnet: bool = True
    live_trading: bool = False
    require_live_confirmation: bool = True
    live_confirmation_env: str = "BOT_CONFIRM_LIVE"
    poll_interval_seconds: int = 30
    lookback_limit: int = 720
    paper_account_balance: float = 1_000.0
    top_symbols: int = 5
    portfolio_metric: str = "profit_factor"
    testnet_base_url: str = "https://demo-fapi.binance.com"
    live_base_url: str = "https://fapi.binance.com"
    demo_account_asset: str = "USDT"
    balance_refresh_seconds: int = 30
    learning_window: int = 50
    use_rl_policy: bool = False
    use_optimizer_output: bool = False
    use_learning_store: bool = False
    max_margin_utilization: float = 0.9
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"


@dataclass(frozen=True)
class SizingConfig:
    mode: str = "atr"
    atr_period: int = 14
    atr_multiple: float = 1.0
    std_window: int = 20
    std_multiple: float = 1.0
    min_notional: float = 5.0
    max_notional: float | None = None


@dataclass(frozen=True)
class MultiTimeframeConfig:
    enabled: bool = False
    confirm_timeframes: Tuple[str, ...] = ("5m",)
    ema_fast: int = 21
    ema_slow: int = 55
    rsi_upper: float = 65
    rsi_lower: float = 35
    require_alignment: bool = True


@dataclass(frozen=True)
class ExternalSignalConfig:
    enabled: bool = False
    sentiment_weight: float = 0.4
    news_weight: float = 0.35
    onchain_weight: float = 0.25
    suppression_threshold: float = -0.2
    boost_threshold: float = 0.4


@dataclass(frozen=True)
class ReinforcementConfig:
    reward_alpha: float = 0.6
    penalty_alpha: float = 0.4
    regime_window: int = 30
    loss_streak_threshold: int = 4
    win_streak_threshold: int = 4
    volatility_threshold: float = 1.5
    min_trades_before_update: int = 20
    update_cooldown_seconds: int = 600
    hard_parameter_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "atr_stop": (1.0, 5.0),
            "atr_target": (1.2, 6.0),
            "rsi_overbought": (45.0, 85.0),
            "rsi_oversold": (15.0, 55.0),
            "fast_ema": (5.0, 80.0),
            "slow_ema": (20.0, 220.0),
            "cooldown_bars": (0.0, 30.0),
            "hold_bars": (20.0, 400.0),
        }
    )


@dataclass(frozen=True)
class RLConfig:
    enabled: bool = True
    apply_to_live: bool = False
    max_param_deviation_from_baseline: float = 0.3
    lookback_window: int = 60
    reward_scheme: str = "risk_adjusted"
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 1e-3
    checkpoint_dir_name: str = "rl_checkpoints"
    episodes: int = 200
    checkpoint_interval: int = 25
    max_steps_per_episode: int = 1_000
    device_preference: str = "auto"
    validation_episodes: int = 5
    results_dir_name: str = "rl_runs"
    policy_dir_name: str = "rl_policies"
    min_validation_reward: float = 0.0


@dataclass(frozen=True)
class BotConfig:
    paths: PathConfig
    universe: UniverseConfig
    risk: RiskConfig
    backtest: BacktestConfig
    strategy: StrategyConfig
    optimizer: OptimizerConfig
    runtime: RuntimeConfig
    sizing: SizingConfig
    multi_timeframe: MultiTimeframeConfig
    external_signals: ExternalSignalConfig
    reinforcement: ReinforcementConfig
    rl: RLConfig


def ensure_directories(paths: PathConfig, extra: Iterable[Path] | None = None) -> None:
    """Create the common directory structure if it does not already exist."""
    targets = [paths.data_dir, paths.results_dir, paths.optimization_dir, paths.log_dir]
    if extra:
        targets.extend(extra)
    for folder in targets:
        folder.mkdir(parents=True, exist_ok=True)


def _validate_risk_config(payload: RiskConfig) -> None:
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ConfigValidationError(message)

    _require(0 < payload.per_trade_risk <= 1, "per_trade_risk must be between 0 and 1")
    _require(payload.leverage > 0, "leverage must be positive")
    _require(payload.min_free_margin >= 0, "min_free_margin must be non-negative")
    _require(0 < payload.max_account_exposure <= 1, "max_account_exposure must be in (0, 1]")
    _require(payload.max_symbol_exposure >= 0, "max_symbol_exposure must be non-negative")
    _require(payload.max_symbol_exposure <= payload.max_account_exposure, "max_symbol_exposure cannot exceed max_account_exposure")
    _require(payload.per_trade_risk <= payload.max_account_exposure, "per_trade_risk cannot exceed max_account_exposure")
    _require(0 <= payload.margin_buffer < 1, "margin_buffer must be in [0, 1)")
    _require(payload.margin_warning_cooldown >= 0, "margin_warning_cooldown must be non-negative")
    _require(payload.margin_relief_factor >= 1.0, "margin_relief_factor must be >= 1")
    _require(payload.max_concurrent_symbols >= 0, "max_concurrent_symbols cannot be negative")
    _require(0 <= payload.max_daily_loss_pct <= 1, "max_daily_loss_pct must be between 0 and 1")
    if payload.max_daily_loss_abs is not None:
        _require(payload.max_daily_loss_abs > 0, "max_daily_loss_abs must be positive when set")
    _require(payload.daily_loss_lookback_hours >= 1, "daily_loss_lookback_hours must be >= 1")


def _validate_runtime_config(runtime: RuntimeConfig) -> None:
    if runtime.poll_interval_seconds <= 0:
        raise ConfigValidationError("poll_interval_seconds must be positive")
    if runtime.lookback_limit <= 0:
        raise ConfigValidationError("lookback_limit must be positive")
    if runtime.balance_refresh_seconds <= 0:
        raise ConfigValidationError("balance_refresh_seconds must be positive")
    if runtime.max_margin_utilization <= 0 or runtime.max_margin_utilization > 1:
        raise ConfigValidationError("max_margin_utilization must be within (0, 1]")


def _validate_rl_config(rl: RLConfig) -> None:
    if not (0 <= rl.max_param_deviation_from_baseline <= 1):
        raise ConfigValidationError("max_param_deviation_from_baseline must be within [0, 1]")
    if rl.episodes <= 0 or rl.max_steps_per_episode <= 0:
        raise ConfigValidationError("RL training episodes and steps must be positive")
    if rl.validation_episodes < 0:
        raise ConfigValidationError("validation_episodes cannot be negative")


def _validate_bot_config(config: BotConfig) -> None:
    _validate_risk_config(config.risk)
    _validate_runtime_config(config.runtime)
    _validate_rl_config(config.rl)


def _build_paths(base_dir: Path | None) -> PathConfig:
    base = base_dir or Path(_env_str("BOT_BASE_DIR", str(Path(__file__).resolve().parents[2]))).resolve()
    data_dir = base / "data"
    results_dir = base / "results"
    optimization_dir = base / "optimization_results"
    log_dir = base / "logs"
    return PathConfig(base, data_dir, results_dir, optimization_dir, log_dir)


def _apply_demo_live_overrides(risk: RiskConfig, runtime: RuntimeConfig) -> Tuple[RiskConfig, RuntimeConfig]:
    runtime = replace(runtime, dry_run=False, use_testnet=True, live_trading=True, max_margin_utilization=min(runtime.max_margin_utilization, 0.85))
    risk = replace(risk, per_trade_risk=min(risk.per_trade_risk, 0.02), leverage=min(risk.leverage, 10.0))
    return risk, runtime


def load_config(*, base_dir: Path | None = None) -> BotConfig:
    """Load a fresh BotConfig instance with optional environment overrides."""
    paths = _build_paths(base_dir)
    universe = UniverseConfig(
        timeframes=_env_tuple("BOT_TIMEFRAMES", UniverseConfig().timeframes),
        demo_symbols=_env_tuple("BOT_DEMO_SYMBOLS", UniverseConfig().demo_symbols),
        default_symbols=_env_tuple("BOT_DEFAULT_SYMBOLS", UniverseConfig().default_symbols),
    )

    runtime_default = RuntimeConfig()
    runtime = RuntimeConfig(
        dry_run=_env_bool("BOT_DRY_RUN", runtime_default.dry_run),
        use_testnet=_env_bool("BOT_USE_TESTNET", runtime_default.use_testnet),
        live_trading=_env_bool("BOT_LIVE_TRADING", runtime_default.live_trading),
        require_live_confirmation=_env_bool("BOT_REQUIRE_LIVE_CONFIRMATION", runtime_default.require_live_confirmation),
        live_confirmation_env=_env_str("BOT_LIVE_CONFIRM_ENV", runtime_default.live_confirmation_env),
        poll_interval_seconds=_env_int("BOT_POLL_INTERVAL", runtime_default.poll_interval_seconds),
        lookback_limit=_env_int("BOT_LOOKBACK_LIMIT", runtime_default.lookback_limit),
        paper_account_balance=_env_float("BOT_PAPER_BALANCE", runtime_default.paper_account_balance),
        top_symbols=_env_int("BOT_TOP_SYMBOLS", runtime_default.top_symbols),
        portfolio_metric=_env_str("BOT_PORTFOLIO_METRIC", runtime_default.portfolio_metric),
        testnet_base_url=_env_str("BOT_TESTNET_BASE_URL", runtime_default.testnet_base_url),
        live_base_url=_env_str("BOT_LIVE_BASE_URL", runtime_default.live_base_url),
        demo_account_asset=_env_str("BOT_DEMO_ASSET", runtime_default.demo_account_asset).upper(),
        balance_refresh_seconds=_env_int("BOT_BALANCE_REFRESH_SECONDS", runtime_default.balance_refresh_seconds),
        learning_window=_env_int("BOT_LEARNING_WINDOW", runtime_default.learning_window),
        use_rl_policy=_env_bool("BOT_USE_RL_POLICY", runtime_default.use_rl_policy),
        use_optimizer_output=_env_bool("BOT_USE_OPTIMIZER_OUTPUT", runtime_default.use_optimizer_output),
        use_learning_store=_env_bool("BOT_USE_LEARNING_STORE", runtime_default.use_learning_store),
        max_margin_utilization=_env_float("BOT_MAX_MARGIN_UTILIZATION", runtime_default.max_margin_utilization),
        api_host=_env_str("BOT_API_HOST", runtime_default.api_host),
        api_port=_env_int("BOT_API_PORT", runtime_default.api_port),
        log_level=_env_str("BOT_LOG_LEVEL", runtime_default.log_level),
    )

    risk_default = RiskConfig()
    risk = RiskConfig(
        per_trade_risk=_env_float("BOT_PER_TRADE_RISK", risk_default.per_trade_risk),
        leverage=_env_float("BOT_LEVERAGE", risk_default.leverage),
        taker_fee=_env_float("BOT_TAKER_FEE", risk_default.taker_fee),
        maker_fee=_env_float("BOT_MAKER_FEE", risk_default.maker_fee),
        min_free_margin=_env_float("BOT_MIN_FREE_MARGIN", risk_default.min_free_margin),
        margin_buffer=_env_float("BOT_MARGIN_BUFFER", risk_default.margin_buffer),
        margin_warning_cooldown=_env_float("BOT_MARGIN_WARNING_COOLDOWN", risk_default.margin_warning_cooldown),
        margin_relief_factor=_env_float("BOT_MARGIN_RELIEF_FACTOR", risk_default.margin_relief_factor),
        max_symbol_exposure=_env_float("BOT_MAX_SYMBOL_EXPOSURE", risk_default.max_symbol_exposure),
        max_account_exposure=_env_float("BOT_MAX_ACCOUNT_EXPOSURE", risk_default.max_account_exposure),
        max_concurrent_symbols=_env_int("BOT_MAX_CONCURRENT_SYMBOLS", risk_default.max_concurrent_symbols),
        max_daily_loss_pct=_env_float("BOT_MAX_DAILY_LOSS_PCT", risk_default.max_daily_loss_pct),
        max_daily_loss_abs=_env_float("BOT_MAX_DAILY_LOSS_ABS", risk_default.max_daily_loss_abs) if os.getenv("BOT_MAX_DAILY_LOSS_ABS") else risk_default.max_daily_loss_abs,
        stop_trading_on_daily_loss=_env_bool("BOT_STOP_ON_DAILY_LOSS", risk_default.stop_trading_on_daily_loss),
        close_positions_on_daily_loss=_env_bool("BOT_CLOSE_ON_DAILY_LOSS", risk_default.close_positions_on_daily_loss),
        daily_loss_lookback_hours=_env_int("BOT_DAILY_LOSS_LOOKBACK", risk_default.daily_loss_lookback_hours),
    )

    rl_default = RLConfig()
    rl = RLConfig(
        enabled=_env_bool("BOT_RL_ENABLED", rl_default.enabled),
        apply_to_live=_env_bool("BOT_RL_APPLY_TO_LIVE", rl_default.apply_to_live),
        max_param_deviation_from_baseline=_env_float("BOT_RL_MAX_DEVIATION", rl_default.max_param_deviation_from_baseline),
        lookback_window=_env_int("BOT_RL_LOOKBACK", rl_default.lookback_window),
        reward_scheme=_env_str("BOT_RL_REWARD_SCHEME", rl_default.reward_scheme),
        gamma=_env_float("BOT_RL_GAMMA", rl_default.gamma),
        entropy_coef=_env_float("BOT_RL_ENTROPY", rl_default.entropy_coef),
        value_coef=_env_float("BOT_RL_VALUE_COEF", rl_default.value_coef),
        learning_rate=_env_float("BOT_RL_LR", rl_default.learning_rate),
        checkpoint_dir_name=_env_str("BOT_RL_CHECKPOINT_DIR", rl_default.checkpoint_dir_name),
        episodes=_env_int("BOT_RL_EPISODES", rl_default.episodes),
        checkpoint_interval=_env_int("BOT_RL_CHECKPOINT_INTERVAL", rl_default.checkpoint_interval),
        max_steps_per_episode=_env_int("BOT_RL_MAX_STEPS", rl_default.max_steps_per_episode),
        device_preference=_env_str("BOT_RL_DEVICE", rl_default.device_preference),
        validation_episodes=_env_int("BOT_RL_VALIDATION_EPISODES", rl_default.validation_episodes),
        results_dir_name=_env_str("BOT_RL_RESULTS_DIR", rl_default.results_dir_name),
        policy_dir_name=_env_str("BOT_RL_POLICY_DIR", rl_default.policy_dir_name),
        min_validation_reward=_env_float("BOT_RL_MIN_VAL_REWARD", rl_default.min_validation_reward),
    )

    backtest = BacktestConfig()
    strategy = StrategyConfig()
    optimizer = OptimizerConfig()
    sizing = SizingConfig()
    multi_timeframe = MultiTimeframeConfig()
    external_signals = ExternalSignalConfig()
    reinforcement = ReinforcementConfig()

    if _env_bool("BOT_ENABLE_DEMO_LIVE", False):
        risk, runtime = _apply_demo_live_overrides(risk, runtime)

    bot_config = BotConfig(
        paths=paths,
        universe=universe,
        risk=risk,
        backtest=backtest,
        strategy=strategy,
        optimizer=optimizer,
        runtime=runtime,
        sizing=sizing,
        multi_timeframe=multi_timeframe,
        external_signals=external_signals,
        reinforcement=reinforcement,
        rl=rl,
    )
    _validate_bot_config(bot_config)
    return bot_config


__all__ = [
    "BotConfig",
    "BacktestConfig",
    "ConfigValidationError",
    "ExternalSignalConfig",
    "MultiTimeframeConfig",
    "OptimizerConfig",
    "PathConfig",
    "ReinforcementConfig",
    "RiskConfig",
    "RLConfig",
    "RuntimeConfig",
    "SizingConfig",
    "StrategyConfig",
    "UniverseConfig",
    "ensure_directories",
    "load_config",
]
