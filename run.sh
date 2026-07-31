#!/usr/bin/env bash
# Lianghua Quant 启动脚本（Git Bash / Linux / macOS）
set -e
cd "$(dirname "$0")"

# 选一个可用的 python：优先 $LIANGHUA_PYTHON，其次本地 venv，最后 PATH。
# 具体哪个解释器装了 streamlit 由 start.py 自行判定，这里只负责把它拉起来。
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
  echo "[错误] 找不到 python，请先安装 Python 3.10+ 或设置 LIANGHUA_PYTHON=<解释器路径>" >&2
  exit 1
}

echo "启动 Lianghua Quant 多资产量化终端..."
exec "$PY" start.py "$@"
