@echo off
chcp 65001 >nul
cd /d %~dp0

rem 一键启动：真实行情后端(8600) + 量化终端(8510)
rem 终端用 8510 而非 8501，规避与 StockSignal / 其它量化软件的端口冲突。
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
    pause
    exit /b 1
)

echo [1/2] 启动真实行情后端 (http://localhost:8600) ...
start "Lianghua 行情后端" /min "%PY%" backend/data_server.py --live-poll 30
timeout /t 2 >nul

echo [2/2] 启动量化终端 (http://localhost:8510) ...
echo        关闭本窗口即停止终端；行情后端请手动关闭其独立窗口。
"%PY%" -m streamlit run lianghua/ui/app.py --server.port 8510 --theme.base dark
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
