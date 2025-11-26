import json
import os
from datetime import datetime
from typing import Any, Dict

DATA_DIR = os.path.join("data", "results")
os.makedirs(DATA_DIR, exist_ok=True)


def save_backtest_results(results: Dict[str, Any]):
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = os.path.join(DATA_DIR, f"backtest_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return path
