#!/usr/bin/env bash
# Lianghua Quant 一键启动：真实行情后端(8600) + 量化终端(8510)
# 终端用 8510 规避与 StockSignal / 其它量化软件的 8501 端口冲突。
set -e
cd "$(dirname "$0")"

pick_python() {
  local c
  for c in "$LIANGHUA_PYTHON" ./.venv/bin/python ./.venv/Scripts/python.exe \
           ./venv/bin/python ./venv/Scripts/python.exe; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  for c in python3 python; do
    command -v "$c" >/dev/null 2>&1 && { command -v "$c"; return 0; }
  done
  return 1
}

PY="$(pick_python)" || {
  echo "[错误] 找不到 python，请先安装 Python 3.10+ 或设置 LIANGHUA_PYTHON" >&2
  exit 1
}

echo "[1/2] 启动真实行情后端 (http://localhost:8600) ..."
"$PY" backend/data_server.py --live-poll 30 >/tmp/lianghua_backend.log 2>&1 &
BACK_PID=$!
sleep 2

echo "[2/2] 启动量化终端 (http://localhost:8510) ..."
"$PY" -m streamlit run lianghua/ui/app.py --server.port 8510 --theme.base dark

# 终端退出时顺手停掉后端
kill "$BACK_PID" 2>/dev/null || true
