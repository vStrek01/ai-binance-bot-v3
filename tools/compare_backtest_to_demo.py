from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .log_utils import DEFAULT_LOG_PATH, coerce_timestamp, iter_log_events


def load_demo_equity(log_path: Path, symbol: Optional[str]) -> pd.Series:
    records = []
    for event in iter_log_events(log_path):
        if event.event != "EQUITY_SNAPSHOT":
            continue
        if symbol and event.symbol and event.symbol != symbol:
            continue
        ts = coerce_timestamp(event.payload.get("timestamp") or event.recorded_ts)
        equity = event.payload.get("equity") or event.payload.get("balance")
        if ts and equity is not None:
            try:
                records.append((ts, float(equity)))
            except (TypeError, ValueError):
                continue
    if not records:
        raise RuntimeError(f"No EQUITY_SNAPSHOT entries found in {log_path}")
    series = pd.Series({idx: value for idx, value in records}, name="demo_equity")
    return series.sort_index()


def load_backtest_equity(csv_path: Path, symbol: Optional[str]) -> pd.Series:
    df = pd.read_csv(csv_path)
    ts_col = _detect_column(df.columns, {"timestamp", "time", "date"})
    equity_col = _detect_column(df.columns, {"equity", "balance", "net_value"})
    symbol_col = _detect_column(df.columns, {"symbol", "pair"}, required=False)
    if not ts_col or not equity_col:
        raise RuntimeError("Backtest CSV needs timestamp and equity columns")
    if symbol and symbol_col:
        df = df[df[symbol_col] == symbol]
    df = df.dropna(subset=[ts_col, equity_col])
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col)
    df.set_index(ts_col, inplace=True)
    return df[equity_col].rename("backtest_equity")


def _detect_column(columns, candidates, required: bool = True) -> Optional[str]:
    columns_lower = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns_lower:
            return columns_lower[candidate]
    for col in columns:
        for candidate in candidates:
            if candidate in col.lower():
                return col
    if required:
        return None
    return None


def compute_stats(series: pd.Series) -> Tuple[float, float]:
    clean = series.dropna()
    if clean.empty:
        return (0.0, 0.0)
    start = clean.iloc[0]
    end = clean.iloc[-1]
    pct = ((end - start) / start) * 100 if start else 0.0
    max_drawdown = ((clean / clean.cummax()) - 1.0).min() * 100
    return pct, max_drawdown


def compare(backtest: pd.Series, demo: pd.Series) -> pd.DataFrame:
    combined = pd.concat([backtest, demo], axis=1).sort_index().interpolate(method="time")
    combined["delta"] = combined["demo_equity"] - combined["backtest_equity"]
    combined["delta_pct"] = combined["delta"] / combined["backtest_equity"] * 100
    return combined


def maybe_plot(df: pd.DataFrame, output: Optional[Path]) -> None:
    if not output:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print("matplotlib missing, skipping plot")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["backtest_equity"], label="Backtest")
    ax.plot(df.index, df["demo_equity"], label="Demo")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backtest CSV equity vs demo log equity snapshots")
    parser.add_argument("backtest_csv", type=Path, help="CSV file with equity column")
    parser.add_argument("--demo-log", type=Path, default=DEFAULT_LOG_PATH, help="Path to logs/bot.log")
    parser.add_argument("--symbol", help="Optional symbol filter")
    parser.add_argument("--plot", type=Path, help="Optional PNG output path")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Alert threshold in percent divergence")
    parser.add_argument("--rows", type=int, default=20, help="Rows from start/end to print")
    args = parser.parse_args()

    demo = load_demo_equity(args.demo_log, args.symbol)
    backtest = load_backtest_equity(args.backtest_csv, args.symbol)
    combined = compare(backtest, demo)
    maybe_plot(combined, args.plot)

    backtest_growth, backtest_dd = compute_stats(combined["backtest_equity"])
    demo_growth, demo_dd = compute_stats(combined["demo_equity"])
    max_abs = combined["delta_pct"].abs().max()

    print("Backtest Return:  %.2f%%" % backtest_growth)
    print("Demo Return:      %.2f%%" % demo_growth)
    print("Backtest DD:      %.2f%%" % backtest_dd)
    print("Demo DD:          %.2f%%" % demo_dd)
    print("Max divergence:   %.2f%%" % max_abs)
    if max_abs > args.tolerance:
        print(f"WARNING: divergence exceeds {args.tolerance}% tolerance")

    head = combined.head(args.rows // 2)
    tail = combined.tail(args.rows // 2)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print("\nStart sample:")
        print(head)
        print("\nEnd sample:")
        print(tail)


if __name__ == "__main__":
    main()
