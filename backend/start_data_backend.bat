@echo off
rem Launcher for the real-data backend (backend/data_server.py).
rem
rem ASCII-only on purpose: `chcp 65001` together with multi-byte CJK text makes
rem cmd.exe lose track of its byte offsets and garble the lines that follow.
rem Reuses the same interpreter-picking logic as the project's run.bat.

cd /d %~dp0

set "PY="
if defined LIANGHUA_PYTHON if exist "%LIANGHUA_PYTHON%" set "PY=%LIANGHUA_PYTHON%"
if not defined PY if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
if not defined PY if exist "..\venv\Scripts\python.exe" set "PY=..\venv\Scripts\python.exe"
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

echo Starting Lianghua real-data backend (default port 8600) ...
echo Ctrl+C to stop.
"%PY%" data_server.py %*
rem keep the real exit code: a bare "pause" would reset ERRORLEVEL to 0
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
