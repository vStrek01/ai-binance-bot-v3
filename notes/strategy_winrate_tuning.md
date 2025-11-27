# Strategy Win-Rate Tuning

This playbook captures the knobs introduced for higher-quality EMA/RSI/ATR signals plus the quick checks to verify them before deployment.

## Filter Overview
- **Expanded parameter space** – `config.yaml::strategy.parameter_space` now sweeps fast/slow EMAs, RSI bands, ATR stop/targets, cooldown/hold bars, spacing, spread/volatility caps, and higher-timeframe confirmation windows.
- **ATR guardrails** – `baseline_strategy.min_atr_pct` / `max_atr_pct` map into `min_volatility_pct` / `max_volatility_pct`, blocking both chop (< floor) and panic (> ceiling).
- **Spread + session filters** – `max_spread_pct` clamps noisy bars while `baseline_strategy.no_trade_sessions` (UTC by default) defines blackout windows.
- **Higher timeframe bias** – `baseline_strategy.higher_tf_trend_interval` converts into EMA confirmation bars so the 1m strategy can respect the configured higher-TF trend without extra data fetches.

## Verification Checklist
1. **Schema + defaults** – run `pytest tests/test_config_schema.py` to confirm the config surface stays valid.
2. **Signal monitor sanity** – inspect recent candles and snapshots before changing overrides.
3. **Parameter sweep ranking** – run the grid search with the new `--score-metric` option for the metric you care about (total_pnl, win_rate, avg_r).
4. **Backtest or dry-run** – replay `bot.runner backtest` (or `dry-run`) for the target symbol/interval before applying overrides live.

## Reference Commands (PowerShell)
```powershell
cd C:/Users/Anwender/Desktop/ai-binance-bot-v3

# 1) Schema regression
./.venv/Scripts/python.exe -m pytest tests/test_config_schema.py

# 2) Signal monitor snapshot (adjust symbol/interval/limit as needed)
./.venv/Scripts/python.exe scripts/signal_monitor.py --symbol BTCUSDT --interval 1m --limit 1440

# 3) Parameter sweep with win-rate focus
./.venv/Scripts/python.exe scripts/param_sweep.py --symbol BTCUSDT --interval 1m --limit 1440 --score-metric win_rate --top 15

# 4) Full backtest of the latest overrides
./.venv/Scripts/python.exe -m bot.runner backtest --symbol BTCUSDT --interval 1m
```

Document each run’s summary (metrics, overrides applied, veto logs) in `results/` so the next tuning cycle has an auditable baseline.