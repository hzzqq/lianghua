#!/usr/bin/env bash
# Lianghua Quant 启动脚本（Git Bash / Linux / macOS）
set -e
cd "$(dirname "$0")"
echo "启动 Lianghua Quant 多资产量化终端..."
python start.py "$@"
