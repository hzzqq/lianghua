@echo off
rem One-click launch: real-data backend (8600) + quant terminal (8510).
rem
rem ASCII-only on purpose: `chcp 65001` together with multi-byte CJK text makes
rem cmd.exe lose track of its byte offsets and garble the lines that follow,
rem which is exactly how this file used to break on double-click.
rem
rem Terminal listens on 8510 instead of 8501 to avoid port conflicts with
rem StockSignal and other quant tools.

cd /d %~dp0

set "PY="
if defined LIANGHUA_PYTHON if exist "%LIANGHUA_PYTHON%" set "PY=%LIANGHUA_PYTHON%"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe" set "PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo [error] Python not found.
    echo         Set LIANGHUA_PYTHON to your interpreter path.
    pause
    exit /b 1
)

echo [1/2] Starting real-data backend (http://localhost:8600) ...
start "Lianghua Data Backend" /min "%PY%" backend/data_server.py --live-poll 30
timeout /t 2 >nul

echo [2/2] Starting quant terminal (http://localhost:8510) ...
echo        Closing this window stops the terminal.
echo        To stop everything: python tools\stop_services.py
"%PY%" -m streamlit run lianghua/ui/app.py --server.port 8510 --theme.base dark
rem keep the real exit code: a bare "pause" would reset ERRORLEVEL to 0
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
