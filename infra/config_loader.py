import os
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when configuration or environment validation fails."""


class ConfigLoader:
    def __init__(self, path: str = "config.yaml"):
        load_dotenv()
        self.path = path

    def load(self, *, mode_override: Optional[str] = None) -> Dict[str, Any]:
        config = self._read_config_file()
        config.setdefault("live_trading_enabled", False)

        run_mode = self._resolve_run_mode(config, mode_override)
        config["run_mode"] = run_mode
        config["binance"] = self._build_binance_section(config, run_mode)
        config["mode_flags"] = {
            "backtest": run_mode == "backtest",
            "paper": run_mode == "paper",
            "live": run_mode == "live",
        }

        self._apply_defaults(config)
        return config

    def _read_config_file(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _apply_defaults(self, config: Dict[str, Any]) -> None:
        config.setdefault("symbol", "BTCUSDT")
        config.setdefault("strategy_mode", "llm")
        config.setdefault("baseline_strategy", {})
        config["baseline_strategy"].setdefault("ma_length", 50)
        config["baseline_strategy"].setdefault("rsi_length", 14)
        config["baseline_strategy"].setdefault("rsi_oversold", 30)
        config["baseline_strategy"].setdefault("rsi_overbought", 70)
        config["baseline_strategy"].setdefault("size_usd", 1_000.0)
        config["baseline_strategy"].setdefault("stop_loss_pct", 0.01)
        config["baseline_strategy"].setdefault("take_profit_pct", 0.02)

        config.setdefault("risk", {})
        config["risk"].setdefault("max_symbol_notional_usd", 5_000.0)
        config["risk"].setdefault("min_order_notional_usd", 10.0)

        config.setdefault("safety", {})
        config["safety"].setdefault("max_daily_drawdown_pct", 5.0)
        config["safety"].setdefault("max_total_notional_usd", 25_000.0)
        config["safety"].setdefault("max_consecutive_losses", 3)

    def _resolve_run_mode(self, config: Dict[str, Any], override: Optional[str]) -> str:
        candidate = override or os.getenv("RUN_MODE") or config.get("run_mode") or "backtest"
        run_mode = candidate.lower().strip()
        if run_mode not in {"backtest", "paper", "live"}:
            raise ConfigError(f"Unsupported RUN_MODE '{candidate}'. Choose backtest, paper, or live.")
        return run_mode

    def _build_binance_section(self, config: Dict[str, Any], run_mode: str) -> Dict[str, Any]:
        existing = config.get("binance", {})
        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        default_testnet = run_mode != "live"
        explicit_testnet = os.getenv("BINANCE_TESTNET")
        alias_testnet = os.getenv("BOT_USE_TESTNET")
        raw_testnet_flag = explicit_testnet if explicit_testnet not in {None, ""} else alias_testnet
        is_testnet = self._parse_testnet_flag(raw_testnet_flag, default_testnet)

        if run_mode in {"paper", "live"}:
            if not api_key or not api_secret:
                raise ConfigError("BINANCE_API_KEY and BINANCE_API_SECRET are required for paper/live modes")

        if run_mode == "paper" and not is_testnet:
            raise ConfigError("Paper mode requires BINANCE_TESTNET=1 (testnet)")

        if run_mode == "live":
            if is_testnet:
                raise ConfigError("Live mode requires BINANCE_TESTNET=0 (mainnet)")
            confirmation = os.getenv("CONFIRM_LIVE", "")
            if confirmation != "YES_I_UNDERSTAND_THE_RISK":
                raise ConfigError("CONFIRM_LIVE must be set to YES_I_UNDERSTAND_THE_RISK for live trading")
            if not config.get("live_trading_enabled"):
                raise ConfigError("Live trading is disabled in config.yaml (set live_trading_enabled: true)")

        section = {**existing}
        section.update(
            {
                "api_key": api_key,
                "api_secret": api_secret,
                "testnet": is_testnet,
            }
        )
        return section

    def _parse_testnet_flag(self, raw: Optional[str], default: bool) -> bool:
        if raw is None or raw.strip() == "":
            return default
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ConfigError("BINANCE_TESTNET must be '1'/'0' or boolean-like text")
