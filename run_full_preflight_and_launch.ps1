$ErrorActionPreference = 'Stop'
Set-Location -Path "C:\Users\Anwender\Desktop\ai-binance-bot-v3"
$python = ".\.venv\Scripts\python.exe"
if (-not $env:BOT_CONFIG_PATH) {
    $env:BOT_CONFIG_PATH = "config_scalping_demo.yaml"
}

Write-Host "=== Running status-store tests ==="
& $python -m pytest tests/test_status_store.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "status_store pytest failed. Aborting launch."
    exit 1
}

Write-Host "=== BTCUSDT signal_monitor (historical) ==="
& $python -m scripts.signal_monitor --symbol BTCUSDT --interval 1m --limit 720 --source historical
if ($LASTEXITCODE -ne 0) {
    Write-Error "BTCUSDT signal_monitor failed. Aborting launch."
    exit 1
}

Write-Host "=== ETHUSDT signal_monitor (historical) ==="
& $python -m scripts.signal_monitor --symbol ETHUSDT --interval 1m --limit 720 --source historical
if ($LASTEXITCODE -ne 0) {
    Write-Error "ETHUSDT signal_monitor failed. Aborting launch."
    exit 1
}

Write-Host "=== Dry-run sanity check (BTCUSDT,ETHUSDT) ==="
# Limit cycles if there is a flag for it; otherwise just run briefly
& $python -m bot.runner dry-run --interval 1m --symbols BTCUSDT,ETHUSDT --cycles 2
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dry-run sanity check failed. Aborting launch."
    exit 1
}

Write-Host "=== Starting dashboard API on http://127.0.0.1:8000 ==="
$apiArgs = @(
    "-m",
    "bot.runner",
    "api",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
    "--log-level",
    "info"
)
$apiProcess = Start-Process -NoNewWindow -FilePath $python -ArgumentList $apiArgs -PassThru
Start-Sleep -Seconds 3
if ($apiProcess.HasExited) {
    $exit = $apiProcess.ExitCode
    Write-Error "Dashboard API exited early with code $exit. Aborting launch."
    exit 1
}

Write-Host "=== Starting scalping demo-live bot on ALL symbols (config=$($env:BOT_CONFIG_PATH)) (press Ctrl+C to stop) ==="
$botArgs = @(
    "-m",
    "bot.runner",
    "demo-live",
    "--interval",
    "1m",
    "--symbols",
    "ALL"
)
try {
    & $python @botArgs
}
finally {
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Write-Host "Stopping dashboard API (PID $($apiProcess.Id))"
        $apiProcess | Stop-Process
    }
}
