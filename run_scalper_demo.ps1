$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $repoRoot

$venvActivate = Join-Path $repoRoot '.venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    . $venvActivate
}

$defaultSymbols = 'BTCUSDT,ETHUSDT,BCHUSDT,ETCUSDT,LTCUSDT,XRPUSDT'
$symbolsArg = $env:SCALPING_SYMBOLS
if (-not $symbolsArg -or -not $symbolsArg.Trim()) {
    $symbolsArg = $defaultSymbols
    $env:SCALPING_SYMBOLS = $symbolsArg
}

if (-not $env:BOT_CONFIG_PATH) {
    $env:BOT_CONFIG_PATH = 'config_scalping_demo.yaml'
}
$env:SCALPING_PRESET = 'HYPER_AGGRESSIVE'
$env:RUN_MODE = 'demo-live'

Write-Host "[info] Starting demo scalper for symbols: $symbolsArg"
python -m bot.runner demo-live --symbols $symbolsArg --interval 1m
