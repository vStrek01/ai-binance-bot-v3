"""Simple CLI helper for inspecting EMA/RSI/ATR strategy signals."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import mean, median
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot.core.config import BotConfig, load_config
from bot.data import feeds
from bot.signals.sim_utils import SimulatedTrade, simulate_trades
from bot.signals.strategies import EmaRsiAtrStrategy, StrategyParameters, StrategySignal, build_parameters


def _print_effective_params(symbol: str, params: StrategyParameters) -> None:
    core = {
        "fast_ema": params.fast_ema,
        "slow_ema": params.slow_ema,
        "rsi_overbought": params.rsi_overbought,
        "rsi_oversold": params.rsi_oversold,
        "atr_stop": params.atr_stop,
        "atr_target": params.atr_target,
        "cooldown_bars": params.cooldown_bars,
        "hold_bars": params.hold_bars,
        "min_reentry_bars_same_dir": getattr(params, "min_reentry_bars_same_dir", None),
        "min_reentry_bars_flip": getattr(params, "min_reentry_bars_flip", None),
    }
    print(f"effective_strategy_params[{symbol}]: {core}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect strategy signals without placing orders")
    parser.add_argument(
        "--symbol",
        "-s",
        nargs="+",
        default=None,
        help="Symbol(s) to analyze; defaults to cfg.universe.default_symbols",
    )
    parser.add_argument("--interval", default="1m", help="Interval to analyze (default: 1m)")
    parser.add_argument("--limit", type=int, default=2000, help="Number of most recent candles to load")
    parser.add_argument(
        "--source",
        default="historical",
        choices=("historical", "recent"),
        help="historical = cached/downloaded CSVs, recent = direct API fetch",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Optional alternate config path (default: config.yaml in repo root)",
    )
    parser.add_argument("--fast-ema", type=float, default=None, help="Override fast EMA length")
    parser.add_argument("--slow-ema", type=float, default=None, help="Override slow EMA length")
    parser.add_argument("--rsi-overbought", type=float, default=None, help="Override RSI overbought level")
    parser.add_argument("--rsi-oversold", type=float, default=None, help="Override RSI oversold level")
    parser.add_argument("--atr-stop", type=float, default=None, help="Override ATR-based stop multiple")
    parser.add_argument("--atr-target", type=float, default=None, help="Override ATR-based target multiple")
    parser.add_argument("--min-reentry-same", type=int, default=None, help="Override min_reentry_bars_same_dir")
    parser.add_argument("--min-reentry-flip", type=int, default=None, help="Override min_reentry_bars_flip")
    return parser.parse_args()


def _load_cfg(config_path: str | None) -> BotConfig:
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = (REPO_ROOT / config_path).resolve()
        base_dir = cfg_path.parent
    else:
        base_dir = REPO_ROOT
    return load_config(base_dir=base_dir)


def _load_candles_for_symbol(cfg: BotConfig, symbol: str, interval: str, limit: int, source: str) -> pd.DataFrame:
    target_rows = max(limit, 1000)
    if source == "historical":
        frame = feeds.ensure_local_candles(cfg, symbol, interval, min_rows=target_rows)
    elif source == "recent":
        frame = feeds.fetch_recent_candles(cfg, symbol, interval, limit=target_rows)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unsupported source '{source}'")
    trimmed = frame.tail(limit).reset_index(drop=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    trimmed[numeric_cols] = trimmed[numeric_cols].astype(float)
    return trimmed


def _effective_limit(args_limit: int, cfg: BotConfig) -> int:
    requested = int(args_limit)
    runtime_cap = int(getattr(cfg.runtime, "lookback_limit", 2000) or 2000)
    backtest_cap = int(getattr(cfg.backtest, "max_bars", 50000) or 50000)
    api_cap = 1500
    return max(1, min(requested, runtime_cap, backtest_cap, api_cap))


def _format_timestamp(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%d %H:%M")


def _print_signal_line(symbol: str, interval: str, timestamp: str, signal) -> None:
    indicators = signal.indicators
    print(
        f"[{timestamp}] {symbol} {interval} {signal.action} @ {signal.entry_price:.2f} "
        f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} "
        f"rsi={indicators.get('rsi', 0.0):.1f} "
        f"ema_fast={indicators.get('ema_fast', 0.0):.2f} ema_slow={indicators.get('ema_slow', 0.0):.2f} "
        f"atr_pct={indicators.get('atr_pct', 0.0):.2f} spread_pct={indicators.get('spread_pct', 0.0):.2f}"
    )


def _summarize(signals, candles: pd.DataFrame, interval: str, *, requested_limit: int, effective_limit: int) -> None:
    total = len(signals)
    longs = sum(1 for sig in signals if sig.direction == 1)
    shorts = sum(1 for sig in signals if sig.direction == -1)
    bars = len(candles)
    close_times = candles.get("close_time", candles.get("open_time"))
    duration_hours = 0.0
    if close_times is not None and len(close_times) >= 2:
        start = pd.to_datetime(close_times.iloc[0], utc=True)
        end = pd.to_datetime(close_times.iloc[-1], utc=True)
        duration_hours = max((end - start).total_seconds() / 3600.0, 0.0)
    signals_per_hour = (total / duration_hours) if duration_hours > 0 else 0.0
    print(
        f"bars={bars} (requested_limit={requested_limit}, effective_limit={effective_limit}) "
        f"signals={total} (long={longs}, short={shorts}) signals_per_hour={signals_per_hour:.2f}"
    )


def run_for_symbol(symbol: str, cfg: BotConfig, interval: str, limit: int, source: str, overrides: Optional[dict[str, float | int]] = None) -> tuple[pd.DataFrame, List[StrategySignal], List[SimulatedTrade], Optional[dict]]:
    candles = _load_candles_for_symbol(cfg, symbol, interval, limit, source)
    params = build_parameters(cfg, symbol=symbol, overrides=overrides or None)
    _print_effective_params(symbol, params)
    strategy = EmaRsiAtrStrategy(params, symbol=symbol, interval=interval, run_mode=cfg.run_mode)
    signals = strategy.generate_signals(candles)
    trades = simulate_trades(candles, signals)
    snapshot = strategy.latest_snapshot()
    return candles, signals, trades, snapshot


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config)
    limit = _effective_limit(args.limit, cfg)
    overrides: dict[str, float | int] = {}
    if args.fast_ema is not None:
        overrides["fast_ema"] = args.fast_ema
    if args.slow_ema is not None:
        overrides["slow_ema"] = args.slow_ema
    if args.rsi_overbought is not None:
        overrides["rsi_overbought"] = args.rsi_overbought
    if args.rsi_oversold is not None:
        overrides["rsi_oversold"] = args.rsi_oversold
    if args.atr_stop is not None:
        overrides["atr_stop"] = args.atr_stop
    if args.atr_target is not None:
        overrides["atr_target"] = args.atr_target
    if args.min_reentry_same is not None:
        overrides["min_reentry_bars_same_dir"] = int(args.min_reentry_same)
    if args.min_reentry_flip is not None:
        overrides["min_reentry_bars_flip"] = int(args.min_reentry_flip)

    if args.symbol:
        symbols = args.symbol
    else:
        defaults = list(getattr(cfg.universe, "default_symbols", []) or [])
        symbols = defaults if defaults else list(cfg.symbols or [])
    # dedupe while preserving order
    seen = set()
    ordered_symbols = []
    for sym in symbols:
        sym_upper = sym.upper()
        if sym_upper not in seen:
            seen.add(sym_upper)
            ordered_symbols.append(sym_upper)
    symbol_stats = []

    for symbol in ordered_symbols:
        candles, signals, trades, snapshot = run_for_symbol(
            symbol,
            cfg,
            args.interval,
            limit,
            args.source,
            overrides=overrides or None,
        )
        close_times = candles.get("close_time", candles.get("open_time"))
        for signal in signals:
            timestamp_value = close_times.iloc[signal.index] if close_times is not None else None
            ts_label = _format_timestamp(timestamp_value)
            _print_signal_line(symbol, args.interval, ts_label, signal)

        _summarize(signals, candles, args.interval, requested_limit=args.limit, effective_limit=limit)
        if snapshot:
            print("latest_snapshot:")
            for key in sorted(snapshot):
                print(f"  {key}: {snapshot[key]}")

        _print_pnl_summary(trades)

        wins = sum(1 for t in trades if t.outcome == "TP")
        losses = sum(1 for t in trades if t.outcome == "SL")
        flats = sum(1 for t in trades if t.outcome == "FLAT")
        resolved = max(1, wins + losses)
        win_rate = wins / resolved * 100.0
        total_trades = len(trades)
        cum_r = sum(t.r_multiple for t in trades)
        avg_r = cum_r / total_trades if total_trades else 0.0
        symbol_stats.append(
            {
                "symbol": symbol,
                "bars": len(candles),
                "trades": total_trades,
                "wins": wins,
                "losses": losses,
                "flats": flats,
                "win_rate": win_rate,
                "cum_r": cum_r,
                "avg_r": avg_r,
            }
        )

    _print_symbol_table(symbol_stats)

    if overrides:
        print(f"parameter_overrides={overrides}")


def _print_pnl_summary(trades: List[SimulatedTrade]) -> None:
    print("=== PnL summary ===")
    total = len(trades)
    wins = sum(1 for trade in trades if trade.outcome == "TP")
    losses = sum(1 for trade in trades if trade.outcome == "SL")
    flats = sum(1 for trade in trades if trade.outcome == "FLAT")
    resolved = max(1, wins + losses)
    win_rate = wins / resolved * 100

    r_values = [trade.r_multiple for trade in trades]
    avg_r = mean(r_values) if r_values else 0.0
    median_r = median(r_values) if r_values else 0.0
    best_r = max(r_values) if r_values else 0.0
    worst_r = min(r_values) if r_values else 0.0
    cum_r = sum(r_values)

    if r_values:
        equity = np.cumsum(r_values)
        peaks = np.maximum.accumulate(equity)
        drawdowns = equity - peaks
        max_dd = float(drawdowns.min())
    else:
        max_dd = 0.0

    print(
        f"trades={total} (wins={wins}, losses={losses}, flats={flats})"
    )
    print(
        f"win_rate={win_rate:.1f}%  avg_R={avg_r:.2f}  median_R={median_r:.2f}  "
        f"best_R={best_r:.2f}  worst_R={worst_r:.2f}"
    )
    print(f"cum_R={cum_r:.2f}  max_drawdown_R={max_dd:.2f}")

    if trades:
        print("sample trades:")
        for trade in trades[:5]:
            direction = "LONG" if trade.direction == 1 else "SHORT"
            print(
                f"[{trade.open_time:%Y-%m-%d %H:%M}] {direction} entry={trade.entry:.2f} "
                f"stop={trade.stop:.2f} target={trade.target:.2f} exit={trade.close:.2f} "
                f"outcome={trade.outcome} R={trade.r_multiple:.2f}"
            )


def _print_symbol_table(rows: List[dict]) -> None:
    if not rows:
        return
    print("=== Per-symbol PnL (R units) ===")
    header = f"{'symbol':<8} {'bars':>6} {'trades':>7} {'wins':>6} {'losses':>8} {'flats':>6} {'win_rate%':>11} {'cum_R':>8} {'avg_R':>8}"
    print(header)
    for row in rows:
        print(
            f"{row['symbol']:<8} {row['bars']:>6} {row['trades']:>7} {row['wins']:>6} {row['losses']:>8} {row['flats']:>6} "
            f"{row['win_rate']:>10.1f} {row['cum_r']:>8.2f} {row['avg_r']:>8.2f}"
        )


if __name__ == "__main__":
    main()
