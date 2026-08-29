#!/usr/bin/env python
"""Lianghua Quant · 项目启动器。

一键启动多资产量化终端（Streamlit）。会自动挑选一个装有 streamlit 的 Python 解释器。

用法：
    python start.py                 # 默认 http://localhost:8501（仅本机可访问）
    python start.py --port 9000     # 指定端口
    python start.py --no-browser    # 无头模式（服务器部署）
    python start.py --host 0.0.0.0  # 允许局域网内其它设备访问

提示：如使用本机虚拟环境，请先激活（如 venv 的 Scripts/activate），
或直接使用虚拟环境的 python 运行本脚本。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import time
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(ROOT, "lianghua", "ui", "app.py")

_HOME = os.path.expanduser("~")


def _candidate_pythons() -> list[str]:
    """按优先级列出可能装有 streamlit 的解释器（去重、只保留真实存在的）。"""
    cands = [
        # WorkBuddy 受管 venv（不写死用户名，跟随当前账户的 HOME）
        os.path.join(_HOME, ".workbuddy", "binaries", "python", "envs",
                     "default", "Scripts", "python.exe"),
        os.path.join(_HOME, ".workbuddy", "binaries", "python", "envs",
                     "default", "bin", "python"),
        # 项目自带虚拟环境
        os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
        os.path.join(ROOT, ".venv", "bin", "python"),
        os.path.join(ROOT, "venv", "Scripts", "python.exe"),
        os.path.join(ROOT, "venv", "bin", "python"),
    ]
    cands += [p for p in (shutil.which("python3"), shutil.which("python")) if p]

    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        key = os.path.normcase(os.path.abspath(c))
        if key in seen or not os.path.isfile(c):
            continue
        seen.add(key)
        out.append(c)
    return out


def _has_streamlit(py: str) -> bool:
    """真实探测某个解释器能否 import streamlit（而不是只看文件在不在）。"""
    try:
        r = subprocess.run(
            [py, "-c", "import streamlit"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _resolve_python() -> str | None:
    """返回一个确实装有 streamlit 的解释器；找不到返回 None。"""
    if importlib.util.find_spec("streamlit") is not None:
        return sys.executable
    # 当前解释器已确认没有 streamlit，跳过重复探测
    self_key = os.path.normcase(os.path.abspath(sys.executable))
    for py in _candidate_pythons():
        if os.path.normcase(os.path.abspath(py)) == self_key:
            continue
        if _has_streamlit(py):
            return py
    return None


def _port_in_use(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((probe_host, port)) == 0


# ---- 端口互斥锁：确保同一端口只被「本项目」占用，杜绝被 StockSignal 等其它
#      8501 服务抢占后误判为「自己已在运行」而偷偷打开别人的页面。 ----
LOCKFILE = os.path.join(ROOT, "terminal.lock")


def _read_lock() -> dict | None:
    try:
        with open(LOCKFILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_lock(port: int, pid: int) -> None:
    with open(LOCKFILE, "w", encoding="utf-8") as f:
        json.dump({"port": port, "pid": pid, "app": "lianghua"}, f)


def _clear_lock() -> None:
    try:
        os.remove(LOCKFILE)
    except OSError:
        pass


def _port_owned_by_us(port: int) -> bool:
    """锁里记录的端口与当前一致，且该 PID 进程（含其子孙）确实在监听此端口。"""
    lock = _read_lock()
    if not lock or lock.get("port") != port:
        return False
    pid = lock.get("pid")
    if not pid:
        return False
    try:
        import psutil  # 可选依赖；没有就退化为「仅看端口占用」
    except ImportError:
        return True
    try:
        root = psutil.Process(pid)
    except (psutil.Error, OSError):
        return False
    # 收集进程树（本进程 + 所有子孙），streamlit 的监听 socket 在子进程上
    members = [root] + root.children(recursive=True)
    listening = {
        c.laddr.port
        for p in members
        for c in p.net_connections(kind="tcp")
        if c.status == "LISTEN" and c.laddr
    }
    return port in listening


def _open_browser(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def _start_daemon(py: str, args) -> int:
    """后台（守护）模式：启动一个脱离当前控制台的 Streamlit 独立进程。

    - 关闭命令行窗口不会影响服务（进程属于新进程组/新会话，不共享控制台）；
    - 日志追加写 logs/terminal.log，PID 记录在项目根 terminal.pid；
    - 端口就绪后自动打开浏览器。
    """
    log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "terminal.log")
    logf = open(log_path, "a", encoding="utf-8", errors="replace")

    cmd = [
        py, "-m", "streamlit", "run", APP,
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true",
    ]
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": logf, "stderr": logf}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    pid = proc.pid
    with open(os.path.join(ROOT, "terminal.pid"), "w", encoding="utf-8") as f:
        f.write(str(pid))
    _write_lock(args.port, pid)

    url = f"http://localhost:{args.port}"
    print(f"🚀 量化终端已作为后台服务启动 (PID {pid})")
    print(f"   地址: {url}")
    print(f"   日志: logs/terminal.log")
    print(f"   关闭本窗口不影响服务；停止服务请运行「停止量化终端.bat」。")

    # 等待端口就绪后自动打开浏览器
    deadline = time.time() + 20
    while time.time() < deadline:
        if _port_in_use(args.host, args.port):
            _open_browser(url)
            break
        time.sleep(0.5)
    else:
        print(f"[警告] 20 秒内端口 {args.port} 未就绪，请查看 logs/terminal.log。")
    return 0


def _ensure_backend(py: str, port: int = 8600, poll: float = 30) -> bool:
    """「启动文件之后连接后端」：若真实行情后端未就绪，自动拉起后置服务。

    - 用原始 TCP 探测 127.0.0.1:<port> 是否监听（不依赖 urllib，沙箱/代理环境更稳）；
    - 未就绪则以脱离控制台的方式拉起 backend/data_server.py（随本机会话存活，
      关闭启动窗口不影响）；
    - 等待最多 15s 直到端口可连；超时也不阻塞终端启动（终端会以降级模式运行）。
    返回 True 表示后端最终可用。
    """
    def _port_open() -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                return s.connect_ex(("127.0.0.1", port)) == 0
        except OSError:
            return False

    if _port_open():
        print(f"✅ 真实行情后端已在线 (http://localhost:{port})")
        return True

    log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logf = open(os.path.join(log_dir, "backend.log"), "a",
                encoding="utf-8", errors="replace")
    backend_py = os.path.join(ROOT, "backend", "data_server.py")
    # --timeout 控制单次真实源取数超时：网络可达时 akshare/baostock 通常 <1s 返回，
    # 断网/源不可达时快速降级演示（而非长时间卡死），避免慢源把终端取数请求拖超时。
    cmd = [py, backend_py, "--live-poll", str(poll), "--port", str(port),
           "--timeout", "5"]
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": logf, "stderr": logf}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **kwargs)
        with open(os.path.join(ROOT, "backend.pid"), "w", encoding="utf-8") as f:
            f.write(str(proc.pid))
        print(f"🚀 已自动拉起真实行情后端 (PID {proc.pid}) → http://localhost:{port}")
    except Exception as e:
        print(f"[警告] 后端自动拉起失败: {e}（终端将以降级模式运行）")
        return False

    deadline = time.time() + 15
    while time.time() < deadline:
        if _port_open():
            print(f"✅ 真实行情后端就绪 (http://localhost:{port})")
            return True
        time.sleep(0.5)
    print(f"[提示] 后端 15 秒内未就绪，终端仍以降级模式启动；"
          f"稍后页面会展示真实连接状态。")
    return False


def main() -> int:
    # 启动器不该因为控制台编码（如 cp936 下输出 emoji）而崩掉
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):
            pass

    p = argparse.ArgumentParser(description="Lianghua Quant 启动器")
    p.add_argument("--port", type=int, default=8501, help="Streamlit 端口 (默认 8501)")
    p.add_argument("--host", default="127.0.0.1",
                   help="监听地址 (默认 127.0.0.1 仅本机；用 0.0.0.0 开放局域网)")
    p.add_argument("--no-browser", action="store_true", help="无头模式，不尝试打开浏览器")
    p.add_argument("--daemon", action="store_true",
                   help="后台模式：以脱离控制台的独立进程运行，关闭命令行窗口不影响服务")
    args = p.parse_args()

    if not os.path.exists(APP):
        print(f"[错误] 找不到 UI 入口: {APP}")
        return 1

    py = _resolve_python()
    if py is None:
        print("[错误] 没有找到装有 streamlit 的 Python 解释器。")
        print("  请先安装依赖，例如：")
        print(f"    {sys.executable} -m pip install -r "
              f"{os.path.join(ROOT, 'requirements.txt')}")
        print("  或直接用已装 streamlit 的解释器运行： <你的python> start.py")
        return 1

    # 启动文件之后连接后端：终端启动前先确保真实行情后端在线
    _ensure_backend(py)

    if _port_in_use(args.host, args.port):
        # 端口被占用：先判断是不是「我们自己」的 daemon 在跑。
        if args.daemon and _port_owned_by_us(args.port):
            url = f"http://localhost:{args.port}"
            print(f"[提示] 本终端后台服务已在运行 → {url}")
            _open_browser(url)
            return 0
        # 端口被「别人」占用（如 StockSignal 等其它 8501 服务）→ 明确报错、绝不偷偷开浏览器
        print(f"[错误] 端口 {args.port} 已被其它进程占用（不是本终端后台服务）。")
        print(f"  · 请先停止占用该端口的服务（例如 StockSignal），或换端口：")
        print(f"      python start.py --port {args.port + 1}")
        print(f"  · 想知道是谁占用：在 PowerShell 运行")
        print(f"      Get-NetTCPConnection -LocalPort {args.port} -State Listen | Select-Object OwningProcess")
        return 1

    if args.daemon:
        return _start_daemon(py, args)

    cmd = [
        py, "-m", "streamlit", "run", APP,
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true" if args.no_browser else "false",
    ]
    print(f"🚀 启动 Lianghua Quant 多资产终端 → http://localhost:{args.port}")
    if args.host in ("127.0.0.1", "localhost"):
        print("   （仅本机可访问；如需局域网访问请加 --host 0.0.0.0）")
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\n已停止。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
