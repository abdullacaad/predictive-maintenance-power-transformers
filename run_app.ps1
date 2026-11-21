param(
    [int]$Port = 8502,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$AppPath = Join-Path $ProjectRoot "app\streamlit_app.py"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at: $PythonExe"
    exit 1
}

if (-not (Test-Path $AppPath)) {
    Write-Error "Streamlit app not found at: $AppPath"
    exit 1
}

Set-Location $ProjectRoot

$Args = @("-m", "streamlit", "run", $AppPath, "--server.port", "$Port")
if ($Headless) {
    $Args += @("--server.headless", "true")
}

Write-Host "Starting Streamlit app on port $Port..."
& $PythonExe @Args
