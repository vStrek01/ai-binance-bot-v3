# Safety Checklist

Before running any demo-live or live trading mode, walk through every item in this checklist.

1. **Never hard-code keys**
   - Do not create `keys.py` or commit secrets anywhere in the repo.
   - Provide API keys via environment variables (`BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`) or a private `.env` that stays outside version control.
2. **Confirm runtime toggles**
   - Double-check `BOT_USE_TESTNET`, `BOT_ENABLE_DEMO_LIVE`, and related overrides before launching any trading runner.
   - Ensure `config.runtime.dry_run` only flips to `False` when you intentionally opt in to demo-live or live.
3. **Validate risk configuration**
   - Review `config.risk.per_trade_risk`, `config.risk.leverage`, and both `config.risk.max_daily_loss_pct` / `config.risk.max_daily_loss_abs`.
   - Keep daily loss limits aggressive enough that the kill-switch has room to protect the account.
4. **Understand the kill-switch**
   - When cumulative losses exceed the configured percentage or absolute threshold, the RiskEngine halts new entries.
   - If `config.risk.close_positions_on_daily_loss` is enabled, all open positions are flattened automatically once the threshold is breached.
5. **RL policies are experimental**
   - RL overrides stay off unless `BOT_USE_RL_POLICY=1` _and_ `config.rl.enabled=True`.
   - Live/demo-live sessions require `config.rl.apply_to_live=True` (or `BOT_RL_APPLY_TO_LIVE=1`) before overrides are even considered.
   - Guardrails (`config.rl.max_param_deviation_from_baseline`) clamp extreme deviations; monitor the logs/status store for rejection reasons.
6. **Protect the API/dashboard**
   - The FastAPI dashboard binds to `127.0.0.1` by default; keep it local unless you fully secure the environment.
   - If you must expose it (`BOT_API_HOST=0.0.0.0`), place it behind a VPN, SSH tunnel, or reverse proxy that enforces HTTPS and authentication.

Stay paranoid: treat demo-live as a dress rehearsal for real capital and escalate changes slowly.
