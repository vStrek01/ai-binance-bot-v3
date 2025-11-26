# AI Binance Bot v3

A production-grade yet safety-first Binance USD-M futures trading framework. The bot runs the exact same strategy and risk logic across live (testnet by default) and historical backtests to keep decisions consistent and auditable.

## Architecture
```
exchange/          # Binance REST + websockets + order routing
core/              # Models, strategy, LLM adapter, risk, state, trading engine
backtest/          # Execution simulator, runner, metrics
infra/             # Config loader, structured logging, persistence helpers
main.py            # CLI entrypoint (backtest by default)
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

Run a sample backtest (synthetic candles):
```bash
python main.py --mode backtest
```

Run paper or live on Binance (testnet by default):
```bash
# Paper
RUN_MODE=paper BINANCE_API_KEY=... BINANCE_API_SECRET=... python main.py --mode paper

# Live (requires explicit confirms + live_trading_enabled: true)
RUN_MODE=live BINANCE_API_KEY=... BINANCE_API_SECRET=... BINANCE_TESTNET=0 \
	CONFIRM_LIVE=YES_I_UNDERSTAND_THE_RISK python main.py --mode live
```
Live trading requires `live_trading_enabled: true` in `config.yaml`; otherwise the process exits. Never disable `BINANCE_TESTNET` unless you fully understand the risk.

## Testing
```bash
pytest
```

Backtest results are persisted under `data/results/` with equity curves and metrics for later inspection.

## Observability

- `infra/logging.py` exposes `setup_logging()` and `log_event()` so every subsystem emits JSON log lines. Events now cover order lifecycle, position opens/closes, equity snapshots, kill-switch triggers, LLM parse issues, and backtest summaries. Logs stream to stdout by default and (when run via `main.py`) also mirror to `logs/bot.log` plus `logs/dashboard_state.json` for lightweight telemetry.
- `frontend/dashboard.html` + `dashboard.js` render a minimal dashboard that polls `logs/dashboard_state.json` for the latest equity snapshot, open positions, and recent events. Serve `frontend/` via any static host (e.g. `python -m http.server 9000`) or open the HTML file directly when running locally.
