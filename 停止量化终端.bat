@echo off
rem Stop the background quant terminal service.
rem Priority: kill the process tree recorded in terminal.pid;
rem fallback: find the listener on port 8501 and stop it.
cd /d %~dp0

set "PIDFILE=terminal.pid"
if exist "%PIDFILE%" (
    set /p PID=<"%PIDFILE%"
    if defined PID (
        taskkill /PID %PID% /T /F >nul 2>nul
        if not errorlevel 1 (
            echo [stopped] quant terminal background service (PID %PID%)
            del "%PIDFILE%" >nul 2>nul
            goto :done
        )
        del "%PIDFILE%" >nul 2>nul
    )
)

powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess; if ($c) { Stop-Process -Id $c -Force -ErrorAction SilentlyContinue; Write-Host ('[stopped] service on port 8501 (PID ' + $c + ')') } else { Write-Host '[info] no service listening on port 8501' }"

:done
pause
