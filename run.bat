@echo off
chcp 65001 >nul
cd /d %~dp0
echo 启动 Lianghua Quant 多资产量化终端...
python start.py
pause
