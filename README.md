# AI Binance Bot v3

Educational USD-M futures research stack with paper trading, adaptive optimization, and an opt-in testnet demo-live execution path. Everything defaults to offline simulation; enabling live traffic requires explicit acknowledgement via `BOT_ENABLE_DEMO_LIVE`.

## Highlights
- Modular architecture: `bot/core`, `bot/data`, `bot/signals`, `bot/risk`, `bot/execution`, `bot/rl`.
- Safety-first demo-live runner that mirrors trades on Binance testnet only.
- Classic optimizer plus a new self-tuning hyperparameter searcher.
- Deep RL pipeline (environment, PyTorch actor-critic agent, trainer, policy store) with CLI entrypoints.
- Rolling TradeLearningStore keeps blending live feedback back into the EMA/RSI/ATR strategy.
- Compatibility shims keep legacy imports working while the refactor marches on.

## Project layout
```
bot/
  core/config.py        # Dataclasses + env toggles (exposed via bot/config.py)
  data/feeds.py         # Kline downloads, caching, validation helpers
  signals/              # Indicators and EMA/RSI/ATR strategy
  risk/                 # Position sizing, multi-timeframe + external gates
  execution/            # Dry-run + demo-live runners and exchange metadata
  rl/                   # Env, rewards, models, agent, trainer, policy store
  optimization/         # HyperparameterOptimizer (self-tuning wrapper)
  optimizer.py          # Baseline grid/random search
  runner.py             # CLI entry point
  ... (backtester, portfolio, learning, utils, etc.)
bot/core/secrets.py     # Environment-based secret helpers (BINANCE_API_KEY, etc.)
requirements.txt        # Includes PyTorch for RL training
```

## CLI overview
| Command | Purpose |
| --- | --- |
| `download` | Pull historical klines into `data/` |
| `backtest` | Run a single backtest on cached candles |
| `optimize` | Evaluate configurable parameter grid for selected markets |
| `train-all` | Batch optimizer across `config.DEFAULT_SYMBOLS` + `config.TIMEFRAMES` |
| `self-tune` | Multi-round HyperparameterOptimizer that shrinks search ranges between rounds |
| `train-rl` | Train the PyTorch actor-critic agent and persist RL-derived parameters |
| `dry-run` / `dry-run-portfolio` | Paper-trading runners (manual or optimizer-ranked portfolios) |
| `demo-live` | Testnet execution loop (requires `BOT_ENABLE_DEMO_LIVE=1` or `config.runtime.live_trading=True`) |
| `api` | Launch FastAPI dashboard via uvicorn |
| `full-cycle` | Download -> validate -> backtest -> optimize -> multi-symbol dry-run |

Example workflow:
```powershell
python -m bot.runner download --symbols BTCUSDT ETHUSDT --intervals 1m 5m --limit 1500
python -m bot.runner backtest --symbol BTCUSDT --interval 1m --use-best
python -m bot.runner optimize --symbol BTCUSDT --interval 1m
python -m bot.runner self-tune --symbol BTCUSDT --interval 1m --rounds 3 --top 5
python -m bot.runner train-rl --symbol BTCUSDT --interval 1m --episodes 300 --device auto
python -m bot.runner dry-run --symbols ALL --interval 1m --use-best
```

## Deep RL subsystem
- `bot/rl/env.py`: `FuturesTradingEnv` normalizes local candles, adds EMA/RSI/ATR context, tracks PnL/drawdown, and emits derived strategy parameters after each training run.
- `bot/rl/models.py` + `bot/rl/agents.py`: PyTorch actor-critic networks with entropy/value balancing and gradient clipping.
- `bot/rl/trainer.py`: Episode-driven trainer with checkpointing, logging, and automatic persistence to `RLPolicyStore`.
- `bot/rl/policy_store.py`: Atomic JSON store at `optimization_results/rl_policies.json`. CLI or demo-live modes can opt-in via `BOT_USE_RL_POLICY=1`, which makes the runner prefer RL-derived overrides before falling back to optimizer outputs.
- Guardrails live in `config.rl`: set `enabled=False` or omit `BOT_USE_RL_POLICY` to disable overrides entirely, require `apply_to_live=True` (plus `BOT_RL_APPLY_TO_LIVE=1` if desired) before RL parameters are ever applied during live/demo trading, and cap deviations via `max_param_deviation_from_baseline` (defaults to 30%). Any violation is logged and surfaced through the status store as "RL overrides blocked".

Train via:
```powershell
python -m bot.runner train-rl --symbol BTCUSDT --interval 1m --episodes 400 --checkpoint-interval 50 --device cuda
```
Checkpoints land in `data/rl_checkpoints`. Derived strategy knobs (EMA lengths, RSI bands, ATR stops/targets) sync to the policy store and can be consumed by dry-run/demo-live once `BOT_USE_RL_POLICY=1` is set.

## Adaptive optimization
`bot/optimization/hyper.py` implements a coarse-to-fine search:
1. Copy the baseline parameter grid from `StrategyConfig.parameter_space`.
2. Run the standard `Optimizer` and keep the top-N performers.
3. Shrink/shift each parameter range around the winners (with variance-based spreads) and run the next round.
4. Persist aggregated results to `optimization_results/hyper_params.json` so `--use-best` can load them automatically.

`Optimizer` also supports incremental tweaks:
- `optimizer.search_mode="random"` + `random_subset` + `random_seed` samples a deterministic slice of the parameter grid.
- `optimizer.score_metric` chooses which metric drives ranking/persistence (defaults to `total_pnl`).
- `optimizer.min_improvement` + `optimizer.early_stop_patience` halt sequential runs once no combo beats the incumbent by the requested delta for `patience` evaluations.

Use cases:
```powershell
python -m bot.runner self-tune --symbol BTCUSDT --interval 1m --rounds 4 --top 8
```

## Risk + signal stack
- **Risk model:** 0.5% capital allocation per trade with a conservative 3x leverage target, daily loss kill switch (percentage or absolute) that halts trading and surfaces status via the dashboard, plus exchange filters enforcing `min_notional/min_qty/step_size` and default 4 bps taker fees (see `bot/core/config.py`).
- **Signals:** EMA crossover with RSI gating, ATR stops/targets, optional cooldown & hold bars. Live runners pull parameters from (in order) RL policy store (if enabled), optimizer outputs when `--use-best` is passed, TradeLearningStore results, then defaults.
- **Risk helpers:** `bot/risk/position.py` and friends supply ATR/std-dev sizing, multi-timeframe confirmation, and external sentiment/on-chain gates.
- **Data:** `bot/data/feeds.py` handles downloads, caching, validation, and refresh thresholds via `ensure_local_candles`.

## Backtester realism controls
- `config.backtest.realism_level` toggles bundled profiles: `toy` halves friction, `standard` mirrors live defaults, `aggressive` forces larger slippage/fees plus funding & latency penalties.
- `config.backtest.slippage_bps` and `fee_model` set the base execution cost before realism multipliers are applied.
- Set `config.backtest.enable_funding_costs=True` (or rely on the aggressive preset) to apply periodic funding debits/credits using `funding_rate_bps` and `funding_interval_hours`.
- Enable `config.backtest.enable_latency` to nudge entries toward unfavorable intrabar extremes using `latency_ms` to size the delay.
- All realism hooks piggyback on the same sizing/risk logic as paper/live runners, so daily loss halts, exposure caps, and MultiTimeframe/External signal gates remain in play.
- Every backtest response now exports an `equity_curve` plus a `realism` metadata blob (active profile, fee/slippage multipliers, latency/funding flags) so downstream tooling can display or persist richer diagnostics without rehydrating config.
- Prefer `PortfolioBacktester` when you want to replay multiple symbols/timeframes as one allocation: feed it `PortfolioSlice` objects, and it will split the capital base, merge the trades chronologically, and emit aggregate portfolio metrics.

## Demo-live on Binance testnet
```powershell
$Env:BOT_ENABLE_DEMO_LIVE = 1
python -m bot.runner demo-live --symbols ALL --interval 1m --use-best
```
- `BOT_ENABLE_DEMO_LIVE=1` flips `runtime.dry_run` off, forces testnet usage, and keeps a strict margin/risk profile. When you later point at real funds, set `BOT_USE_TESTNET=0`, `BOT_LIVE_TRADING=1`, and export `BOT_CONFIRM_LIVE=1` (or change `runtime.live_confirmation_env`) to acknowledge the risk explicitly. Without the confirmation flag the client factory refuses to create a live client.
- Never touches mainnet; `_client(testnet=True)` points to `config.runtime.testnet_base_url`.
- `_demo_symbol_universe` queries exchangeInfo to discover eligible USDT perpetuals, falling back to `config.DEMO_SYMBOLS` if needed.
- `TradeLearningStore` ingests each closed trade, adapts parameters, and persists them to `optimization_results/learned_params.json`.
- Live runner enforces sizing, leverage, fees, cooldowns, multi-timeframe/external gates, and logs fills to `logs/`.

## ⚠ API & Dashboard Safety
- The FastAPI dashboard binds to `127.0.0.1` by default; set `BOT_API_HOST` deliberately if you must expose it.
- Never listen on `0.0.0.0` without an authenticated tunnel, VPN, or reverse proxy—remote access means remote trading control.
- Use TLS and auth whenever traffic leaves localhost, and rotate keys/secrets immediately if exposure is suspected.
- Review [docs/SAFETY.md](docs/SAFETY.md) before enabling demo-live or live toggles.

## Requirements
- Python 3.10+
- `pip install -r requirements.txt` (includes `torch` for RL training)
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` exported in your shell or stored in a private `.env`

Windows quick start:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m bot.runner download --symbols BTCUSDT --intervals 1m
```

## Resetting adaptive stores
- Delete `optimization_results/rl_policies.json` to clear RL overrides.
- Delete `optimization_results/learned_params.json` + `logs/learned_trades.jsonl` to reset TradeLearningStore history.
- Delete `optimization_results/hyper_params.json` to remove self-tuning outputs.

## Next steps
Before running demo-live or live modes, read [docs/SAFETY.md](docs/SAFETY.md).

1. Add new signal modules (breakouts, volatility filters, depth-of-book features) under `bot/signals/`.
2. Extend RL agent with curriculum learning or PPO-style updates and wire policy evaluation into CLI.
3. Add websocket streaming for faster demo-live fills.
4. Expand dashboard metrics (Calmar, Sortino, per-symbol attribution) and expose RL performance there.
5. Gate live trading behind additional confirmations before pointing at mainnet keys.

Experiment safely and keep everything in simulation until you understand every control knob.
