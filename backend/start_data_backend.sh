#!/usr/bin/env bash
# Lianghua 真实行情后端启动脚本（Git Bash / Linux / macOS）
set -e
cd "$(dirname "$0")"

pick_python() {
  local c
  for c in "$LIANGHUA_PYTHON" ../.venv/bin/python ../.venv/Scripts/python.exe \
           ../venv/bin/python ../venv/Scripts/python.exe; do
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

echo "启动 Lianghua 真实行情后端 (默认端口 8600) ..."
exec "$PY" data_server.py "$@"
