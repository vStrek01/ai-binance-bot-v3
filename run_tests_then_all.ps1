param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "[info] Working directory: $scriptDir"

$venvActivate = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"
$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (Test-Path $venvActivate) {
    Write-Host "[info] Activating .venv..."
    . $venvActivate
} else {
    Write-Host "[warn] .venv\\Scripts\\Activate.ps1 not found; continuing without venv."
}

$testExitCode = 0

if (-not $SkipTests) {
    Write-Host "[info] Running test suite (pytest)..."
    try {
        if (Test-Path $venvPython) {
            & $venvPython -m pytest
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            python -m pytest
        } else {
            pytest
        }
        $testExitCode = $LASTEXITCODE
    } catch {
        Write-Host "[error] Exception while running tests: $_"
        $testExitCode = 1
    }

    if ($testExitCode -ne 0) {
        Write-Host "[error] Tests failed (exit code $testExitCode). Scalper/dashboard will not start."
        exit $testExitCode
    }

    Write-Host "[ok] Tests passed. Launching demo stack..."
} else {
    Write-Host "[info] SkipTests flag set; launching demo stack without running pytest."
}

$runAll = Join-Path $scriptDir "run_all.ps1"
if (-not (Test-Path $runAll)) {
    Write-Host "[error] run_all.ps1 not found in $scriptDir"
    exit 1
}

Write-Host "[info] Invoking run_all.ps1 ..."
& $runAll

exit $testExitCode
