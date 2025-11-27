"""Grid search helper for the EMA/RSI/ATR strategy."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot.core.config import load_config
from bot.signals.strategies import EmaRsiAtrStrategy, build_parameters
from bot.signals.sim_utils import SimulatedTrade, simulate_trades
from scripts.signal_monitor import _effective_limit, _load_candles_for_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parameter sweep for EMA/RSI/ATR strategy.")
    parser.add_argument("--symbol", "-s", default="BTCUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument(
        "--limit", type=int, default=720, help="Number of bars per run (e.g. 720 = 12h of 1m)"
    )
    parser.add_argument("--top", type=int, default=20, help="How many top configs to display")
    parser.add_argument("--config", type=str, default="config.yaml", help="Optional explicit config path")
    parser.add_argument(
        "--score-metric",
        default="total_pnl",
        choices=("total_pnl", "win_rate", "avg_r"),
        help="Ranking metric for the sweep table (default: total_pnl)",
    )
    return parser.parse_args()


def _print_yaml_snippet(symbol: str, overrides: dict[str, float | int]) -> None:
    import textwrap

    lines = [f"{symbol.upper()}:"]
    for key, value in overrides.items():
        if isinstance(value, float):
            rendered = f"{float(value):.1f}"
        else:
            rendered = f"{value}"
        lines.append(f"  {key}: {rendered}")
    snippet = textwrap.indent("\n".join(lines), "  ")
    print("\nYAML snippet for strategy.symbol_overrides:")
    print(snippet)


def _resolve_base_dir(config_path: str | None) -> Path:
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = (REPO_ROOT / config_path).resolve()
        return cfg_path.parent
    return REPO_ROOT


def main() -> None:
    args = parse_args()
    base_dir = _resolve_base_dir(args.config)
    cfg = load_config(base_dir=base_dir)

    symbol = args.symbol.upper()
    effective_limit = _effective_limit(args.limit, cfg)
    df = _load_candles_for_symbol(cfg, symbol, args.interval, effective_limit, "historical")
    if df is None or df.empty:
        print(f"No data for {symbol} {args.interval}")
        return

    fast_ema_vals = [5, 8, 13]
    slow_ema_vals = [13, 21, 34]
    rsi_overbought_vals = [52, 55]
    rsi_oversold_vals = [45, 48]
    atr_stop_vals = [1.4, 1.6, 1.9]
    atr_target_vals = [2.0, 2.4, 2.8]
    min_reentry_same_vals = [8, 12, 16]
    min_reentry_flip_vals = [2, 4]

    grid: list[tuple[int, int, float, float, float, float, int, int]] = []
    for (fast, slow, rsi_hi, rsi_lo, atr_s, atr_t, re_same, re_flip) in itertools.product(
        fast_ema_vals,
        slow_ema_vals,
        rsi_overbought_vals,
        rsi_oversold_vals,
        atr_stop_vals,
        atr_target_vals,
        min_reentry_same_vals,
        min_reentry_flip_vals,
    ):
        if slow <= fast:
            continue
        if rsi_lo >= rsi_hi:
            continue
        grid.append((fast, slow, rsi_hi, rsi_lo, atr_s, atr_t, re_same, re_flip))

    rows: list[dict] = []
    for (fast, slow, rsi_hi, rsi_lo, atr_s, atr_t, re_same, re_flip) in grid:
        overrides = {
            "fast_ema": fast,
            "slow_ema": slow,
            "rsi_overbought": rsi_hi,
            "rsi_oversold": rsi_lo,
            "atr_stop": atr_s,
            "atr_target": atr_t,
            "min_reentry_bars_same_dir": re_same,
            "min_reentry_bars_flip": re_flip,
        }

        params = build_parameters(cfg, symbol=symbol, overrides=overrides)
        strategy = EmaRsiAtrStrategy(params, symbol=symbol, interval=args.interval, run_mode=cfg.run_mode)
        signals = strategy.generate_signals(df)
        trades: List[SimulatedTrade] = simulate_trades(df, signals)

        trades_count = len(trades)
        wins = sum(1 for t in trades if t.outcome == "TP")
        losses = sum(1 for t in trades if t.outcome == "SL")
        flats = sum(1 for t in trades if t.outcome == "FLAT")

        if trades_count:
            r_series = np.array([t.r_multiple for t in trades], dtype=float)
            cum_R = float(r_series.sum())
            avg_R = float(r_series.mean())
            best_R = float(r_series.max())
            worst_R = float(r_series.min())

            equity = r_series.cumsum()
            peak = np.maximum.accumulate(equity)
            drawdowns = equity - peak
            max_dd = float(drawdowns.min())
        else:
            cum_R = avg_R = best_R = worst_R = max_dd = 0.0

        win_rate = wins / max(1, wins + losses) * 100.0
        score = (cum_R / max(1.0, abs(max_dd))) if max_dd < 0 else cum_R

        rows.append(
            {
                "fast_ema": fast,
                "slow_ema": slow,
                "rsi_overbought": rsi_hi,
                "rsi_oversold": rsi_lo,
                "atr_stop": atr_s,
                "atr_target": atr_t,
                "min_reentry_bars_same_dir": re_same,
                "min_reentry_bars_flip": re_flip,
                "trades": trades_count,
                "wins": wins,
                "losses": losses,
                "flats": flats,
                "win_rate": win_rate,
                "cum_R": cum_R,
                "avg_R": avg_R,
                "best_R": best_R,
                "worst_R": worst_R,
                "max_dd": max_dd,
                "score": score,
            }
        )

    if not rows:
        print("No parameter combinations produced trades.")
        return

    results_df = pd.DataFrame(rows)
    metric_map = {
        "total_pnl": "cum_R",
        "win_rate": "win_rate",
        "avg_r": "avg_R",
    }
    primary_metric = metric_map.get(args.score_metric, "cum_R")
    sort_cols = [primary_metric]
    secondary = "cum_R" if primary_metric != "cum_R" else "avg_R"
    sort_cols.append(secondary)
    sort_orders = [False for _ in sort_cols]
    results_df = results_df.sort_values(sort_cols, ascending=sort_orders)
    top_n = min(args.top, len(results_df))
    metric_label = {
        "cum_R": "total_pnl",
        "win_rate": "win_rate",
        "avg_R": "avg_R",
    }[primary_metric]

    print(
        f"=== Top {top_n} configs for {symbol} {args.interval} (limit={effective_limit}, score-metric={metric_label}) ==="
    )
    base_cols = [
        primary_metric,
        "score",
        "cum_R",
        "avg_R",
        "max_dd",
        "trades",
        "win_rate",
        "fast_ema",
        "slow_ema",
        "rsi_overbought",
        "rsi_oversold",
        "atr_stop",
        "atr_target",
        "min_reentry_bars_same_dir",
        "min_reentry_bars_flip",
    ]
    cols: list[str] = []
    for col in base_cols:
        if col not in cols:
            cols.append(col)
    print(results_df[cols].head(top_n).to_string(index=False, float_format=lambda x: f"{x:0.2f}"))

    best = results_df.iloc[0]
    best_overrides = {
        "fast_ema": int(best.fast_ema),
        "slow_ema": int(best.slow_ema),
        "rsi_overbought": float(best.rsi_overbought),
        "rsi_oversold": float(best.rsi_oversold),
        "atr_stop": float(best.atr_stop),
        "atr_target": float(best.atr_target),
        "min_reentry_bars_same_dir": int(best.min_reentry_bars_same_dir),
        "min_reentry_bars_flip": int(best.min_reentry_bars_flip),
    }
    print("\nBest overrides dict:")
    print(best_overrides)
    _print_yaml_snippet(symbol, best_overrides)


if __name__ == "__main__":
    main()
