"""Typed configuration schema shared across runtime components."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RunMode = Literal["backtest", "dry-run", "demo-live", "live"]


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_per_trade_pct: float = Field(1.0, gt=0.0, le=100.0)
    max_daily_drawdown_pct: float = Field(5.0, gt=0.0, le=100.0)
    max_consecutive_losses: int = Field(3, gt=0)
    max_notional_per_symbol: float = Field(5_000.0, gt=0.0)
    max_notional_global: float = Field(25_000.0, gt=0.0)


class ExchangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_testnet: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: str = "https://fapi.binance.com"

    @field_validator("api_key", "api_secret", mode="before")
    @classmethod
    def _strip_empty(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ema_fast: int = Field(13, gt=0)
    ema_slow: int = Field(34, gt=0)
    rsi_length: int = Field(14, gt=0)
    rsi_overbought: int = Field(70, ge=0, le=100)
    rsi_oversold: int = Field(30, ge=0, le=100)
    atr_length: int = Field(14, gt=0)
    atr_multiplier: float = Field(1.5, gt=0.0)


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_confidence: float = Field(1.0, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_mode: RunMode = "backtest"
    risk: RiskConfig = Field(default_factory=lambda: RiskConfig())
    exchange: ExchangeConfig = Field(default_factory=lambda: ExchangeConfig())
    strategy: StrategyConfig = Field(default_factory=lambda: StrategyConfig())
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig())

    @model_validator(mode="after")
    def _validate_live_exchange(self) -> "AppConfig":
        if self.run_mode == "live":
            missing = [name for name in ("api_key", "api_secret") if not getattr(self.exchange, name)]
            if missing:
                raise ValueError("Live mode requires exchange.api_key and exchange.api_secret to be set")
            if self.exchange.use_testnet:
                raise ValueError("Live mode requires exchange.use_testnet to be False")
        return self
