@echo off
cd /d %~dp0
echo 启动 Lianghua Quant 量化终端 (http://localhost:8501) ...
streamlit run lianghua/ui/app.py --server.port 8501
pause
