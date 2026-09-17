#!/usr/bin/env bash
# 停止量化终端 / supervisor / 真实行情后端（与 start_all.sh 对称）。
# 关停逻辑同 Windows 的 停止量化终端.bat，统一由 tools/stop_services.py 提供，
# 因此两端行为不会分叉（先杀 supervisor，避免 streamlit 被 2s 自愈重新拉起）。
set -euo pipefail
cd "$(dirname "$0")"

PY="${LIANGHUA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
  elif [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    PY="python"
  fi
fi

exec "$PY" tools/stop_services.py "$@"
