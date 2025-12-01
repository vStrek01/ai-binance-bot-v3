# Boots the FastAPI dashboard using the demo-live/testnet configuration.
# risk_event and STRATEGY_VETO logs are produced by the trading backend and are not errors.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $repoRoot

$venvActivate = Join-Path $repoRoot '.venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
	. $venvActivate
}

$defaultSymbols = 'BTCUSDT,ETHUSDT,BCHUSDT,ETCUSDT,LTCUSDT,XRPUSDT'
if (-not $env:SCALPING_SYMBOLS -or -not $env:SCALPING_SYMBOLS.Trim()) {
	$env:SCALPING_SYMBOLS = $defaultSymbols
}

if (-not $env:BOT_CONFIG_PATH) {
	$env:BOT_CONFIG_PATH = 'config_scalping_demo.yaml'
}
$env:SCALPING_PRESET = 'HYPER_AGGRESSIVE'

python -m bot.runner api --host 127.0.0.1 --port 8000 --log-level info
