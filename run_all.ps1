$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $repoRoot

$venvActivate = Join-Path $repoRoot '.venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    . $venvActivate
} else {
    Write-Host "[warn] .venv not found; continuing with current interpreter"
}

$defaultSymbols = 'BTCUSDT,ETHUSDT,BCHUSDT,ETCUSDT,LTCUSDT,XRPUSDT'
if (-not $env:BOT_CONFIG_PATH) {
    $env:BOT_CONFIG_PATH = 'config_scalping_demo.yaml'
}
$env:SCALPING_PRESET = 'HYPER_AGGRESSIVE'
if (-not $env:SCALPING_SYMBOLS -or -not $env:SCALPING_SYMBOLS.Trim()) {
    $env:SCALPING_SYMBOLS = $defaultSymbols
}
$env:RUN_MODE = 'demo-live'

$scalperScript = Join-Path $repoRoot 'run_scalper_demo.ps1'
$dashboardScript = Join-Path $repoRoot 'run_dashboard.ps1'

if (-not (Test-Path $scalperScript)) {
    throw "Missing scalper script: $scalperScript"
}
if (-not (Test-Path $dashboardScript)) {
    throw "Missing dashboard script: $dashboardScript"
}

Write-Host "[info] Launching demo scalper for symbols: $env:SCALPING_SYMBOLS"
$scalperArgs = @('-NoLogo','-NoExit','-ExecutionPolicy','Bypass','-File',$scalperScript)
Start-Process -FilePath 'powershell.exe' -ArgumentList $scalperArgs | Out-Null

Write-Host "[info] Launching dashboard server"
$dashboardArgs = @('-NoLogo','-NoExit','-ExecutionPolicy','Bypass','-File',$dashboardScript)
Start-Process -FilePath 'powershell.exe' -ArgumentList $dashboardArgs | Out-Null

Write-Host "[ok] Demo scalper and dashboard started. Leave the new PowerShell windows open to monitor activity."
