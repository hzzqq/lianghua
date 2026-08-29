"""终端守护进程（supervisor）：常驻循环拉起并保活 Streamlit 终端。

为什么需要它：
- 本沙箱会在回合结束时回收「后台启动任务」的启动 shell，
  streamlit 一旦失去父 shell 就会被一并杀掉（HTTP 000）。
- 而常驻阻塞的进程（如后端 data_server 的启动 shell 一直活着）
  不会被回收。本脚本通过「无限循环 + 阻塞 wait」让自己成为
  一个常驻任务，从而：
    1) 不被回合边界回收；
    2) 即使 streamlit 异常退出，也在 2s 后自动重启。

PID 文件（供「停止量化终端.bat」按正确顺序关停）：
- supervisor.pid：supervisor 自身的 PID。**必须先杀它**，否则它会在
  2s 内把 streamlit 重新拉起，终端永远停不掉。
- terminal.pid：当前 streamlit 子进程 PID（每次重启同步更新），
  便于诊断与端口兜底。

用法（由 run_in_background 启动，请勿直接双击）：
    python tools/terminal_supervisor.py
可选环境变量：
    LH_TERMINAL_PORT  端口（默认 8510）
    LH_TERMINAL_HOST  绑定地址（默认 0.0.0.0）
"""
import atexit
import os
import subprocess
import sys
import time

PORT = os.environ.get("LH_TERMINAL_PORT", "8510")
HOST = os.environ.get("LH_TERMINAL_HOST", "0.0.0.0")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PID_FILE = os.path.join(ROOT, "supervisor.pid")
CHILD_PID_FILE = os.path.join(ROOT, "terminal.pid")


def _write_pid(path: str, pid: int) -> None:
    """记录 PID，失败不影响守护主流程（只读诊断用途）。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(pid))
    except OSError:
        pass


def _remove_pid(path: str) -> None:
    """仅在内容确属本进程时删除，避免误删后来者写下的 PID。"""
    try:
        with open(path, encoding="utf-8") as f:
            if f.read().strip() != str(os.getpid()):
                return
        os.remove(path)
    except OSError:
        pass


def _terminate(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """先 terminate，超时再 kill；已退出的进程直接跳过。"""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:                                    # noqa: BLE001
        pass
    try:
        proc.wait(timeout=timeout)
        return
    except Exception:                                    # noqa: BLE001
        pass
    try:
        proc.kill()
        proc.wait(timeout=timeout)
    except Exception:                                    # noqa: BLE001
        pass


def main() -> int:
    os.chdir(ROOT)
    py = sys.executable
    cmd = [
        py, "-m", "streamlit", "run", "lianghua/ui/app.py",
        "--server.port", PORT,
        "--server.headless", "true",
        "--server.address", HOST,
    ]
    _write_pid(PID_FILE, os.getpid())
    atexit.register(_remove_pid, PID_FILE)

    print(f"[supervisor] pid={os.getpid()} -> {PID_FILE}", flush=True)
    print(f"[supervisor] starting terminal on {HOST}:{PORT}", flush=True)

    restart_count = 0
    proc = None
    try:
        while True:
            try:
                proc = subprocess.Popen(cmd)
            except Exception as e:                        # noqa: BLE001
                print(f"[supervisor] failed to spawn streamlit: {e}", flush=True)
                time.sleep(5)
                continue

            _write_pid(CHILD_PID_FILE, proc.pid)
            try:
                rc = proc.wait()
            except KeyboardInterrupt:
                print("[supervisor] interrupted; stopping terminal", flush=True)
                _terminate(proc)
                return 0

            restart_count += 1
            print(
                f"[supervisor] streamlit exited (rc={rc}); "
                f"restart #{restart_count} in 2s",
                flush=True,
            )
            time.sleep(2)
    finally:
        if proc is not None:
            _terminate(proc)


if __name__ == "__main__":
    raise SystemExit(main())
