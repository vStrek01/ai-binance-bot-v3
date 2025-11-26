# AI Binance Bot v3

A production-grade yet safety-first Binance USD-M futures trading framework. The bot runs the exact same strategy and risk logic across live (testnet by default) and historical backtests to keep decisions consistent and auditable.

## Architecture
```
exchange/          # Binance REST + websockets + order routing
core/              # Models, strategy, LLM adapter, risk, state, trading engine
backtest/          # Execution simulator, runner, metrics
infra/             # Config loader, structured logging, persistence helpers
bot/runner.py      # Unified CLI (download/backtest/optimize/dry-run/live/api)
```

Pipeline: **DATA → STRATEGY → LLM (optional) → RISK → ORDER → EXECUTION → STATE**

- **LLM adapter** enforces a strict JSON schema with `action|confidence|reason` only; invalid or unsafe output falls back to `FLAT`.
- **Risk manager** enforces max risk per trade, daily drawdown, open-position limits, leverage caps, and R-based sizing.
- **Order router** sends idempotent client order IDs, attaches stop-loss/take-profit protections, and retries transient failures.
- **Backtester** replays candles through the same strategy + risk stack with spread, slippage, and fees to produce equity curves and metrics.

## Quick start
Install dependencies:
```bash
pip install -r requirements.txt
```

Run a sample backtest (core engine on cached data):
```bash
python -m bot.runner backtest --symbol BTCUSDT --interval 1m
```

Paper trading on the Binance testnet (uses the same strategy + risk stack):
```bash
python -m bot.runner dry-run --symbol BTCUSDT --interval 1m
```

Demo/live trading on the Binance testnet (requires `BOT_LIVE_TRADING=1` and explicit confirmation envs from your config):
```bash
BOT_LIVE_TRADING=1 BOT_CONFIRM_LIVE=YES_I_UNDERSTAND_THE_RISK python -m bot.runner demo-live --symbol BTCUSDT --interval 1m
```
Live trading against production endpoints still requires entering real API keys plus whatever manual confirmations you configure—keep `BOT_API_HOST` at `127.0.0.1` so the observability API is never exposed publicly.

## Testing
```bash
pytest
```

Backtest results are persisted under `data/results/` with equity curves and metrics for later inspection.

## Observability

- `infra/logging.py` now exposes helper accessors (`get_recent_events`, `get_open_positions`, `get_equity_snapshot`) in addition to `setup_logging()`/`log_event()`. Every order/position/equity/killswitch/backtest event is emitted as JSON to stdout and (when configured) mirrored to `logs/dashboard_state.json`.
- `bot.api` surfaces the telemetry over FastAPI: `/api/dashboard/state` returns equity + open positions + recent events while `/api/dashboard/{equity,positions,events}` provide focused slices. The server only listens on `127.0.0.1` unless you override `BOT_API_HOST`.
- `frontend/dashboard.html` + `dashboard.js` now call `/api/dashboard/state`, so serving `frontend/` via `uvicorn bot.api:app --host 127.0.0.1 --port 8000` instantly shows live equity, positions, and recent trades/killswitch events.
- `tools/tail_logs.py` is a lightweight terminal monitor: `python -m tools.tail_logs --interval 1.5` polls the same API endpoint (or `--source local` to read in-process helpers) and prints fresh equity snapshots, open-position deltas, and notable events.

## Experimental & credential stubs

- Anything under `experimental/` (for example `experimental.eval_llm_vs_baseline`) is unsupported and excluded from the production CLI. Run such scripts explicitly via `python -m experimental.<name>`.
- The old `keys_example.py` now lives under `docs/keys_example.py` with a DO NOT IMPORT warning so contributors know which environment variables to configure without risking accidental imports in production code.
- Reinforcement-learning checkpoints/results (`data/rl_checkpoints/`, `data/rl_runs/`) are ignored in git—rerun `python -m bot.runner train-rl ...` to generate your own artifacts locally.
