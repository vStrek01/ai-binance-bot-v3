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

## Demo-Live Trading (Binance Futures Testnet)

`demo-live` mode connects directly to the Binance USDS-M Futures testnet (`demo-fapi` REST + `demo-stream` websockets). Orders are placed against your demo balance via the full production stack—risk engine, kill-switch, trade lifecycle, and strategy logic all run exactly as they would in live trading, but there is **zero real-money risk**.

### Demo API keys
- Visit https://testnet.binancefuture.com (or https://demo.binance.com depending on your region).
- Create a Futures demo API key/secret pair and enable trading permissions.
- Store the credentials securely; never reuse live keys in demo mode.

### Required environment variables
Set the following before launching the bot (replace the placeholder values with your demo credentials):

```bash
BINANCE_API_KEY="dXKBFVDv8uK1YSdzn75BxXRzL6fDJHYtcRGyrCwS8r7K9EAwHCP18ezrKc9lKVxw"
BINANCE_API_SECRET="NuIJw6tYF1XSxbmkoAgdQOjNrXS2CPEcqoRl7yUqJOJQI9OxXcBmODV6jNW9Pcxu"
RUN_MODE=demo-live
```

- Do **not** set the live-trading confirmation env vars (`BOT_CONFIRM_LIVE`, etc.) while in demo-live.
- `use_testnet` must remain `true`; the config loader enforces safe defaults for all demo REST/WS URLs when `RUN_MODE=demo-live`.

### Example `config.yaml`

```yaml
run_mode: demo-live

exchange:
	use_testnet: true
	rest_base_url: "https://demo-fapi.binance.com"
	ws_market_url: "wss://demo-stream.binancefuture.com/stream"
	ws_user_url: "wss://demo-stream.binancefuture.com/ws"
```

You can override strategy, risk, or other sections as needed; leaving the URLs unset lets the loader inject the same safe defaults shown above.

### Starting the bot

```powershell
cd C:/Users/Anwender/Desktop/ai-binance-bot-v3
& ./.venv/Scripts/Activate.ps1

python -m bot.runner `
	--run-mode demo-live `
	--symbol BTCUSDT `
	--interval 1m
```

Expect log lines advertising `run_mode=demo-live`, websocket/REST connections targeting the Binance Futures testnet, order placement against your demo balance, kill-switch/risk limit enforcement, and equity snapshots updating throughout the session.

> **WARNING**
> `demo-live` is not real live trading—funds are virtual, and switching to true live mode requires explicit confirmations plus a different configuration. Double-check your run mode and API keys before moving to production.

## Typed Config Schema

`infra/config_loader.load_config()` now returns an `AppConfig` instance backed by the Pydantic models in `infra/config_schema.py`. YAML, environment variables, and CLI overrides are merged into a single payload and validated before any trading logic initializes. A typical JSON representation looks like:

```json
{
	"run_mode": "dry-run",
	"risk": {
		"risk_per_trade_pct": 1.5,
		"max_daily_drawdown_pct": 6.0,
		"max_consecutive_losses": 4,
		"max_notional_per_symbol": 2500.0,
		"max_notional_global": 10000.0
	},
	"exchange": {
		"use_testnet": true,
		"api_key": null,
		"api_secret": null,
		"base_url": "https://testnet.binancefuture.com"
	},
	"strategy": {
		"ema_fast": 13,
		"ema_slow": 34,
		"rsi_length": 14,
		"rsi_overbought": 70,
		"rsi_oversold": 30,
		"atr_length": 14,
		"atr_multiplier": 1.5
	},
	"llm": {
		"enabled": false,
		"max_confidence": 0.8
	}
}
```

Percentages must fall within 0–100, counts must be positive integers, and notionals must be greater than zero. When `run_mode` is `live`, exchange API keys are mandatory and `use_testnet` must be `false`. Any violation raises a `ConfigError`, so the process fails fast before orders can be submitted.

## Dependencies & CI
- Runtime packages live in `requirements.in`; dev tooling goes into `requirements-dev.in`, which already includes `-r requirements.in` so you only list the extras.
- After editing either `.in` file, regenerate the lockfiles locally with:

```bash
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt
```

- The GitHub Actions workflow installs from `requirements-dev.txt` (compiling it on the fly if the file is empty) before running `pytest`, so keep that lock committed to benefit from pip caching.

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
