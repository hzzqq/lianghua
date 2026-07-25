"""支持 `python -m lianghua` 直接启动多资产量化终端。"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(ROOT, "ui", "app.py")


def main() -> int:
    if not os.path.exists(APP):
        print(f"[错误] 找不到 UI 入口: {APP}")
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", APP, "--server.headless", "false"]
    print(f"🚀 启动 Lianghua Quant 多资产终端 → http://localhost:8501")
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\n已停止。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
