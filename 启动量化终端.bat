@echo off
rem Double-click entry with a Chinese name; delegates to run.bat.
cd /d %~dp0
call "%~dp0run.bat" %*
