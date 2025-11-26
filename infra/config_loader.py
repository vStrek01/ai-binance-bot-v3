import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    def __init__(self, path: str = "config.yaml"):
        load_dotenv()
        self.path = path

    def load(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        config.setdefault("binance", {})
        config["binance"].setdefault("api_key", os.getenv("BINANCE_API_KEY", ""))
        config["binance"].setdefault("api_secret", os.getenv("BINANCE_API_SECRET", ""))
        config.setdefault("live_trading_enabled", False)
        config.setdefault("symbol", "BTCUSDT")
        config.setdefault("testnet", True)
        return config
