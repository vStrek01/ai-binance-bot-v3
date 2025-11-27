# Boots the FastAPI dashboard using the demo-live/testnet configuration.
# risk_event and STRATEGY_VETO logs are produced by the trading backend and are not errors.
$ErrorActionPreference = 'Stop'
Set-Location -Path "C:/Users/Anwender/Desktop/ai-binance-bot-v3"
& "C:/Users/Anwender/Desktop/ai-binance-bot-v3/.venv/Scripts/python.exe" -m bot.runner api --host 127.0.0.1 --port 8000 --log-level info
