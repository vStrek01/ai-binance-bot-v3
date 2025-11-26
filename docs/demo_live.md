# Demo-Live Runbook (Binance Futures Testnet)

> **Audience**: Operators who want the bot to place *real orders* on the Binance Futures **testnet** (demo-fapi). No real funds move in this mode, but the full strategy, risk engine, and kill-switch are active exactly as they would be in production.

## 1. Overview
- `RUN_MODE=demo-live` targets the Binance USDⓈ-M Futures testnet REST (`https://testnet.binancefuture.com`) and streaming (`wss://stream.binancefuture.com`) endpoints.
- The config loader locks testnet URLs, enforces `exchange.use_testnet=True`, and clamps risk defaults (0.1% per trade, 1% daily drawdown, tight notional caps, SL/TP required) for this mode.
- Orders travel through: **strategy → RL/LLM gates (off by default) → risk engine → order manager → Binance UMFutures testnet**.
- Observability (FastAPI + structured logs) tags every event with `run_mode` and the testnet flag so you can audit exactly what happened.

## 2. Requirements
### Binance Futures testnet account & API keys
1. Visit <https://testnet.binancefuture.com> (aka `demo.binance.com`) and create a Futures testnet account.
2. Generate an API key + secret with *trade* permissions. **Never** reuse your production keys.
3. Fund the demo wallet from the Binance faucet if needed.

### Local environment variables (`.env` example)
Create a `.env` (or export in your shell) with **placeholder-style** values — do not paste real-looking credentials into source control:

```
BINANCE_API_KEY=DEMO_KEY_PLACEHOLDER
BINANCE_API_SECRET=DEMO_SECRET_PLACEHOLDER
RUN_MODE=demo-live
BOT_LIVE_TRADING=1
BINANCE_TESTNET=1
```

Notes:
- Leave `BOT_CONFIRM_LIVE` unset; demo-live should not rely on the mainnet confirmation workflow.
- `BINANCE_TESTNET` / `BOT_USE_TESTNET` are redundant when `RUN_MODE=demo-live`, but exporting them keeps the intent explicit.

## 3. Minimal `config.yaml`
The loader already injects safe defaults, but the following snippet shows the canonical settings you should keep for demo-live:

```yaml
run_mode: demo-live

runtime:
  dry_run: false
  live_trading: true
  use_testnet: true
  require_live_confirmation: true
  poll_interval_seconds: 30
  max_margin_utilization: 0.35

exchange:
  use_testnet: true
  rest_base_url: "https://testnet.binancefuture.com"
  ws_market_url: "wss://stream.binancefuture.com/stream"
  ws_user_url: "wss://stream.binancefuture.com/ws"

risk:
  per_trade_risk: 0.001        # 0.1% of equity per trade
  max_daily_loss_pct: 0.01      # halt after 1% drawdown
  leverage: 10                  # enforced via Futures change_leverage
  max_consecutive_losses: 3     # kill-switch streak
  max_notional_per_symbol: 500  # USD cap per symbol
  max_notional_global: 1500     # USD cap across portfolio
  require_sl_tp: true           # enforce stop & target on every entry
```

You can customize strategy parameters, sizing mode, or external signals as needed; the safety-sensitive options above should remain cautious until you graduate to real capital.

## 4. Launch commands (Windows PowerShell)
Run the bot from the repo root with the project virtual environment activated:

```powershell
cd C:/Users/Anwender/Desktop/ai-binance-bot-v3
& ./.venv/Scripts/Activate.ps1

python -m bot.runner `
  demo-live `
  --symbol BTCUSDT `
  --interval 1m `
  --use-best
```

- The CLI banner will log `RUN_MODE=demo-live` along with the REST/WS endpoints and whether RL/LLM overrides remain disabled.
- The FastAPI dashboard (`python -m bot.runner api`) will expose `/api/dashboard/state` with run mode, `use_testnet`, open positions, equity snapshots, and the current kill-switch/risk state.

## 5. Demo-live pre-flight checklist (run before *every* session)
- **.env sanitized** — `BINANCE_API_KEY/SECRET` placeholders replaced locally, `RUN_MODE=demo-live`, `BOT_LIVE_TRADING=1`, `BINANCE_TESTNET=1`, and no real secrets committed.
- **Config audit** — `config.yaml` shows `run_mode=demo-live`, `exchange.use_testnet=true`, URLs contain only `testnet.binancefuture.com` / `stream.binancefuture.com`, and risk caps remain conservative (0.1% per trade, 1% daily drawdown, leverage 10, SL/TP required).
- **Dry-run smoke test** — run the scripted `BOT_DRY_RUN=1` command; verify the banner references demo endpoints, `EQUITY_SNAPSHOT` logs emit, and there are zero `ORDER_*` events.
- **Observability clean** — dashboard/logs report `run_mode=demo-live`, `use_testnet=true`, no `KILL_SWITCH_TRIGGERED`, and prior kill-switch logs (if any) are understood/resolved.
- **RL/LLM posture** — `runtime.use_rl_policy=False` and `llm.enabled=False` unless explicitly reviewed and re-enabled for this session.
- **Network + firewall** — outbound access to `demo-fapi.binance.com`, `demo-stream.binancefuture.com`, and `stream.binancefuture.com` is permitted; local firewall exceptions applied.
- **Demo wallet health** — Futures testnet wallet has sufficient margin; faucet top-up confirmed.
- **Process hygiene** — no lingering bot sessions or background tasks; telemetry storage rotated so the dashboard can write `logs/dashboard_state.json`.
- **Pytest status** — `pytest` suite is green (config loader, runtime gating, time drift, exchange alignment, integration tests).

## 6. Troubleshooting tips
- **Orders blocked (symbol_abs_cap / portfolio_abs_cap)**: your requested size exceeds the demo notional caps; lower the leverage or reduce the portfolio width.
- **"missing_sl_tp" warnings**: the strategy emitted a signal without both stop & target; verify your indicator config and that the timeframe has sufficient ATR/RSI data.
- **Kill-switch triggered**: inspect the `recent_events` list (look for `reason`), confirm the daily drawdown or loss streak, and restart the process once you've investigated.
- **Dashboard empty**: ensure `LOG_EVENT_BUFFER` is set high enough (defaults to 50) and consider exporting `DASHBOARD_STATE_FILE=logs/dashboard_state.json` so the FastAPI layer can persist recent events between restarts.

Stay in demo-live until you have several days of stable execution logs, risk events, and trades recorded. Only then consider enabling true live trading with a separate config + explicit confirmations.
