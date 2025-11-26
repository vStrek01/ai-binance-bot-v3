"""Typed configuration schema shared across runtime components."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RunMode = Literal["backtest", "dry-run", "demo-live", "live"]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_dir: Path
    data_dir: Path
    results_dir: Path
    optimization_dir: Path
    log_dir: Path


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeframes: Tuple[str, ...] = ("1m", "5m", "15m", "1h")
    demo_symbols: Tuple[str, ...] = ("BTCUSDT", "BCHUSDT", "ETHUSDT", "ETCUSDT", "LTCUSDT", "XRPUSDT")
    default_symbols: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "LTCUSDT")


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_trade_risk: float = Field(0.005, gt=0.0, lt=1.0)
    leverage: float = Field(3.0, gt=0.0)
    taker_fee: float = Field(0.0004, ge=0.0)
    maker_fee: float = Field(0.0002, ge=0.0)
    min_free_margin: float = Field(50.0, ge=0.0)
    margin_buffer: float = Field(0.1, ge=0.0, lt=1.0)
    margin_warning_cooldown: float = Field(90.0, ge=0.0)
    margin_relief_factor: float = Field(1.15, ge=1.0)
    max_symbol_exposure: float = Field(0.1, ge=0.0)
    max_account_exposure: float = Field(0.25, gt=0.0, le=1.0)
    max_concurrent_symbols: int = Field(3, ge=0)
    max_daily_loss_pct: float = Field(0.02, ge=0.0, le=1.0)
    max_daily_loss_abs: float | None = Field(default=None, gt=0.0)
    stop_trading_on_daily_loss: bool = True
    close_positions_on_daily_loss: bool = False
    daily_loss_lookback_hours: int = Field(24, ge=1)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_balance: float = Field(1_000.0, gt=0.0)
    fee_model: str = "taker"
    slippage_bps: float = Field(1.0, ge=0.0)
    max_bars: int = Field(50_000, ge=0)
    realism_level: Literal["toy", "standard", "aggressive"] = "standard"
    enable_funding_costs: bool = False
    funding_rate_bps: float = Field(1.0, ge=0.0)
    funding_interval_hours: int = Field(8, ge=1)
    enable_latency: bool = False
    latency_ms: int = Field(250, ge=0)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_parameters: Dict[str, float] = Field(
        default_factory=lambda: {
            "fast_ema": 13.0,
            "slow_ema": 34.0,
            "rsi_length": 14.0,
            "rsi_overbought": 60.0,
            "rsi_oversold": 40.0,
            "atr_period": 14.0,
            "atr_stop": 1.6,
            "atr_target": 2.2,
            "cooldown_bars": 2.0,
            "hold_bars": 90.0,
        }
    )
    parameter_space: Dict[str, List[float]] = Field(
        default_factory=lambda: {
            "fast_ema": [8.0, 13.0, 21.0],
            "slow_ema": [34.0, 55.0, 89.0],
            "rsi_overbought": [58.0, 60.0, 65.0],
            "rsi_oversold": [35.0, 40.0, 45.0],
            "atr_stop": [1.4, 1.6, 1.9],
            "atr_target": [1.8, 2.2, 2.8],
        }
    )


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_parallel: bool = True
    max_workers: int = Field(4, ge=1)
    max_param_combinations: int | None = Field(default=None, ge=1)
    max_symbols_per_run: int | None = Field(default=None, ge=1)
    randomize: bool = False
    search_mode: Literal["grid", "random"] = "grid"
    random_subset: int | None = Field(default=None, ge=1)
    random_seed: int | None = None
    score_metric: str = "total_pnl"
    early_stop_patience: int | None = Field(default=None, ge=1)
    min_improvement: float = Field(0.0, ge=0.0)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dry_run: bool = True
    use_testnet: bool = True
    live_trading: bool = False
    require_live_confirmation: bool = True
    live_confirmation_env: str = "BOT_CONFIRM_LIVE"
    poll_interval_seconds: int = Field(30, ge=1)
    lookback_limit: int = Field(720, ge=1)
    paper_account_balance: float = Field(1_000.0, gt=0.0)
    top_symbols: int = Field(5, ge=1)
    portfolio_metric: str = "profit_factor"
    testnet_base_url: str = "https://demo-fapi.binance.com"
    live_base_url: str = "https://fapi.binance.com"
    demo_account_asset: str = "USDT"
    balance_refresh_seconds: int = Field(30, ge=1)
    learning_window: int = Field(50, ge=1)
    use_rl_policy: bool = False
    use_optimizer_output: bool = False
    use_learning_store: bool = False
    max_margin_utilization: float = Field(0.9, gt=0.0, le=1.0)
    api_host: str = "127.0.0.1"
    api_port: int = Field(8000, ge=1)
    log_level: str = "INFO"


class SizingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "atr"
    atr_period: int = Field(14, ge=1)
    atr_multiple: float = Field(1.0, gt=0.0)
    std_window: int = Field(20, ge=1)
    std_multiple: float = Field(1.0, gt=0.0)
    min_notional: float = Field(5.0, ge=0.0)
    max_notional: float | None = Field(default=None, gt=0.0)


class MultiTimeframeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    confirm_timeframes: Tuple[str, ...] = ("5m",)
    ema_fast: int = Field(21, ge=1)
    ema_slow: int = Field(55, ge=1)
    rsi_upper: float = Field(65.0, ge=0.0, le=100.0)
    rsi_lower: float = Field(35.0, ge=0.0, le=100.0)
    require_alignment: bool = True


class ExternalSignalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sentiment_weight: float = Field(0.4, ge=0.0, le=1.0)
    news_weight: float = Field(0.35, ge=0.0, le=1.0)
    onchain_weight: float = Field(0.25, ge=0.0, le=1.0)
    suppression_threshold: float = -0.2
    boost_threshold: float = 0.4


class ReinforcementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward_alpha: float = Field(0.6, ge=0.0, le=1.0)
    penalty_alpha: float = Field(0.4, ge=0.0, le=1.0)
    regime_window: int = Field(30, ge=1)
    loss_streak_threshold: int = Field(4, ge=0)
    win_streak_threshold: int = Field(4, ge=0)
    volatility_threshold: float = Field(1.5, ge=0.0)
    min_trades_before_update: int = Field(20, ge=0)
    update_cooldown_seconds: int = Field(600, ge=1)
    hard_parameter_bounds: Dict[str, Tuple[float, float]] = Field(
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


class RLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    apply_to_live: bool = False
    max_param_deviation_from_baseline: float = Field(0.3, ge=0.0, le=1.0)
    lookback_window: int = Field(60, ge=1)
    reward_scheme: str = "risk_adjusted"
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    entropy_coef: float = Field(0.01, ge=0.0)
    value_coef: float = Field(0.5, ge=0.0)
    learning_rate: float = Field(1e-3, gt=0.0)
    checkpoint_dir_name: str = "rl_checkpoints"
    episodes: int = Field(200, ge=1)
    checkpoint_interval: int = Field(25, ge=1)
    max_steps_per_episode: int = Field(1_000, ge=1)
    device_preference: str = "auto"
    validation_episodes: int = Field(5, ge=0)
    results_dir_name: str = "rl_runs"
    policy_dir_name: str = "rl_policies"
    min_validation_reward: float = 0.0


class ExchangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_testnet: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rest_base_url: str = "https://fapi.binance.com"
    ws_market_url: str = "wss://fstream.binance.com"
    ws_user_url: Optional[str] = "wss://fstream.binance.com/ws"

    @field_validator("api_key", "api_secret", "ws_user_url", mode="before")
    @classmethod
    def _strip_empty(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_confidence: float = Field(1.0, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_mode: RunMode = "backtest"
    paths: PathsConfig
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    multi_timeframe: MultiTimeframeConfig = Field(default_factory=MultiTimeframeConfig)
    external_signals: ExternalSignalConfig = Field(default_factory=ExternalSignalConfig)
    reinforcement: ReinforcementConfig = Field(default_factory=ReinforcementConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @model_validator(mode="after")
    def _validate_modes(self) -> "AppConfig":
        rest_url = (self.exchange.rest_base_url or "").lower()
        demo_hosts = ("demo-fapi.binance.com", "testnet.binancefuture.com")
        mainnet_hosts = ("fapi.binance.com",)

        if self.run_mode == "demo-live":
            if not self.exchange.use_testnet:
                raise ValueError("Demo-live requires exchange.use_testnet to be True")
            if not any(host in rest_url for host in demo_hosts):
                raise ValueError("Demo-live requires exchange.rest_base_url to point to Binance Futures testnet")

        if self.run_mode == "live":
            missing = [name for name in ("api_key", "api_secret") if not getattr(self.exchange, name)]
            if missing:
                raise ValueError("Live mode requires exchange.api_key and exchange.api_secret to be set")
            if self.exchange.use_testnet:
                raise ValueError("Live mode requires exchange.use_testnet to be False")
            if not any(host in rest_url for host in mainnet_hosts) or any(host in rest_url for host in demo_hosts):
                raise ValueError("Live mode requires exchange.rest_base_url to target Binance Futures mainnet")

        if self.runtime.max_margin_utilization <= 0 or self.runtime.max_margin_utilization > 1:
            raise ValueError("runtime.max_margin_utilization must be within (0, 1]")
        if self.risk.per_trade_risk > self.risk.max_account_exposure:
            raise ValueError("risk.per_trade_risk cannot exceed risk.max_account_exposure")
        if self.risk.max_symbol_exposure > self.risk.max_account_exposure:
            raise ValueError("risk.max_symbol_exposure cannot exceed risk.max_account_exposure")

        return self
