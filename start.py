#!/usr/bin/env python
"""Lianghua Quant · 项目启动器。

一键启动多资产量化终端（Streamlit）。会自动用当前 Python 解释器拉起 streamlit。

用法：
    python start.py                 # 默认 http://localhost:8501
    python start.py --port 9000     # 指定端口
    python start.py --no-browser     # 无头模式（服务器部署）
    python start.py --host 127.0.0.1 # 仅本机访问

提示：如使用本机虚拟环境，请先激活（如 venv 的 Scripts/activate），
或直接使用虚拟环境的 python 运行本脚本。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(ROOT, "lianghua", "ui", "app.py")


def main() -> int:
    p = argparse.ArgumentParser(description="Lianghua Quant 启动器")
    p.add_argument("--port", type=int, default=8501, help="Streamlit 端口 (默认 8501)")
    p.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    p.add_argument("--no-browser", action="store_true", help="无头模式，不尝试打开浏览器")
    args = p.parse_args()

    if not os.path.exists(APP):
        print(f"[错误] 找不到 UI 入口: {APP}")
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run", APP,
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true" if args.no_browser else "false",
    ]
    print(f"🚀 启动 Lianghua Quant 多资产终端 → http://localhost:{args.port}")
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\n已停止。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
