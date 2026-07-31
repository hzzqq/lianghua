@echo off
chcp 65001 >nul
cd /d %~dp0

rem Pick a usable python. start.py itself decides which interpreter has streamlit.
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
    echo         Install Python 3.10+ and add it to PATH,
    echo         or set LIANGHUA_PYTHON to your interpreter path.
    pause
    exit /b 1
)

echo Starting Lianghua Quant terminal ...
"%PY%" start.py %*
rem keep the real exit code: a bare "pause" would reset ERRORLEVEL to 0
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
