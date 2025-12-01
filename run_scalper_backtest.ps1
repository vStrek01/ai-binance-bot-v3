$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $repoRoot

$venvActivate = Join-Path $repoRoot '.venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    . $venvActivate
}

if (-not $env:BOT_CONFIG_PATH) {
    $env:BOT_CONFIG_PATH = 'config_scalping_demo.yaml'
}
$env:SCALPING_PRESET = 'HYPER_AGGRESSIVE'

python -m bot.runner backtest --engine core --symbol BTCUSDT --interval 1m
