# Launches the scalping demo-live bot on Binance Futures Testnet (use_testnet=true, demo-fapi URLs).
# Expect risk_event and STRATEGY_VETO logs as part of normal strategy behavior.
$ErrorActionPreference = 'Stop'
Set-Location -Path "C:/Users/Anwender/Desktop/ai-binance-bot-v3"
if (-not $env:BOT_CONFIG_PATH) {
	$env:BOT_CONFIG_PATH = "config_scalping_demo.yaml"
}
& "C:/Users/Anwender/Desktop/ai-binance-bot-v3/.venv/Scripts/python.exe" -m bot.runner demo-live --interval 1m --symbols ALL
