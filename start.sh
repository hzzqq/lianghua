#!/usr/bin/env bash
# Lianghua Quant 启动脚本（Git Bash / Linux / macOS）
# 与 run.sh 等价，保留此文件名是为了兼容旧的使用习惯。
cd "$(dirname "$0")"
exec bash ./run.sh "$@"
