"""Helpers to project bot risk settings into the shared core risk schema."""
from __future__ import annotations

from bot.core.config import BotConfig
from core.models import RiskConfig as CoreRiskConfig


def build_core_risk_config(cfg: BotConfig) -> CoreRiskConfig:
    """Translate AppConfig risk knobs into the canonical core RiskConfig."""
    slippage = cfg.backtest.slippage_bps / 10_000
    return CoreRiskConfig(
        max_risk_per_trade_pct=float(cfg.risk.per_trade_risk or 0.0),
        max_daily_drawdown_pct=float(cfg.risk.max_daily_loss_pct or 0.0),
        max_open_positions=int(cfg.risk.max_concurrent_symbols or 1),
        max_leverage=int(max(1, round(cfg.risk.leverage))),
        taker_fee_rate=float(cfg.risk.taker_fee or 0.0),
        maker_fee_rate=float(cfg.risk.maker_fee or 0.0),
        slippage=float(slippage),
        max_symbol_notional_usd=float(cfg.risk.max_notional_per_symbol or 0.0),
        max_total_notional_usd=float(cfg.risk.max_notional_global or 0.0),
        min_order_notional_usd=float(cfg.sizing.min_notional or 10.0),
        max_trades_per_day=int(cfg.risk.max_trades_per_day or 0),
        max_commission_pct_per_day=float(cfg.risk.max_commission_pct_per_day or 0.0),
        max_consecutive_losses=int(cfg.risk.max_consecutive_losses or 0),
    )


__all__ = ["build_core_risk_config"]
