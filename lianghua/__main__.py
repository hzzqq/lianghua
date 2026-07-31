"""支持 `python -m lianghua` 直接启动多资产量化终端。"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(ROOT, "ui", "app.py")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):
            pass

    if not os.path.exists(APP):
        print(f"[错误] 找不到 UI 入口: {APP}")
        return 1
    if importlib.util.find_spec("streamlit") is None:
        print("[错误] 当前解释器没有安装 streamlit：")
        print(f"    {sys.executable}")
        print("  请先安装依赖： pip install -r requirements.txt")
        print("  或改用项目根目录的启动器： python start.py（会自动挑选可用解释器）")
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", APP, "--server.headless", "false"]
    print("🚀 启动 Lianghua Quant 多资产终端 → http://localhost:8501")
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\n已停止。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
