@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "APP_PATH=%PROJECT_ROOT%app\streamlit_app.py"
set "DEFAULT_PORT=8502"

if /I "%~1"=="stop" goto :stop
if /I "%~1"=="start" shift

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python executable not found: "%PYTHON_EXE%"
  exit /b 1
)

if not exist "%APP_PATH%" (
  echo [ERROR] Streamlit app not found: "%APP_PATH%"
  exit /b 1
)

echo Starting Streamlit app...
"%PYTHON_EXE%" -m streamlit run "%APP_PATH%" --server.port %DEFAULT_PORT% %*
exit /b %errorlevel%

:stop
set "PORT=%~2"
if "%PORT%"=="" set "PORT=%DEFAULT_PORT%"

echo Stopping Streamlit app on port %PORT%...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$connections = Get-NetTCPConnection -State Listen -LocalPort %PORT% -ErrorAction SilentlyContinue;" ^
  "if (-not $connections) { exit 3 };" ^
  "$pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique;" ^
  "Stop-Process -Id $pids -Force"

if errorlevel 3 (
  echo [INFO] No running Streamlit process found on port %PORT%.
  exit /b 0
)

if errorlevel 1 (
  echo [ERROR] Failed to stop the app on port %PORT%.
  exit /b 1
)

echo Streamlit app stopped on port %PORT%.
exit /b 0
