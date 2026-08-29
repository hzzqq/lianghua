@echo off
chcp 65001 >nul
cd /d %~dp0

rem 真实行情后端启动器：复用项目选 python 逻辑，再拉起 backend/data_server.py
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
    pause
    exit /b 1
)

echo Starting Lianghua 真实行情后端 (默认端口 8600) ...
"%PY%" data_server.py %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
