import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_backtest_results(results: Dict[str, Any], *, symbol: str, interval: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{interval}_backtest_{timestamp}.json"
    path = RESULTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return path
