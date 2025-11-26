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
python main.py --backtest
```

Run live on Binance (testnet by default):
```bash
BINANCE_API_KEY=... BINANCE_API_SECRET=... python main.py --live
```
Live trading requires `live_trading_enabled: true` in `config.yaml`; otherwise the process exits. Set `testnet: false` only when you are certain you want mainnet traffic.

## Testing
```bash
pytest
```

Backtest results are persisted under `data/results/` with equity curves and metrics for later inspection.
