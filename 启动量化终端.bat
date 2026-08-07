@echo off
rem Double-click entry with a Chinese name; delegates to run.bat.
rem Daemon mode (--daemon): run Streamlit as a detached background service
rem that survives this console window. Stop it via the stop script.
cd /d %~dp0
call "%~dp0run.bat" --daemon %*
