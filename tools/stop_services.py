"""停止量化终端 / supervisor / 真实行情后端（跨平台，可单测）。

为什么把停止逻辑从 .bat 挪进 Python：
- **顺序是硬约束**：supervisor 会在 streamlit 被杀后 2s 内自愈重启，
  所以必须**先杀 supervisor**，否则终端永远停不掉。这个顺序在 .bat 里
  依赖 `EnableDelayedExpansion` + `!PID!`，写错静默失效、难以验证。
- **可测试 / 可维护**：Python 版本能被 pytest 直接覆盖（见
  `tests/test_service_lifecycle.py`），.bat 只能人肉双击验证。
- **跨平台一致**：Windows 的 `停止量化终端.bat` 与 Unix 的 `stop_all.sh`
  共用同一份逻辑，不会两边行为分叉。
- **绕开 cmd 编码坑**：中文提示留在 Python 里，.bat 正文保持纯 ASCII。

关停顺序：
    1) supervisor.pid —— 必须先杀，否则自愈重启
    2) terminal.pid   —— streamlit 子进程（未被托管时）
    3) backend.pid    —— 真实行情后端 :8600
    4) 端口兜底        —— 8501 / 8510 / 8600 上仍在监听的残留进程
    5) 清理 terminal.lock（start.py 写入的端口互斥锁，停止后不应残留）

用法：
    python tools/stop_services.py            # 停止全部
    python tools/stop_services.py --dry-run  # 只报告将要做什么
    python tools/stop_services.py --keep-backend  # 保留数据后端
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 顺序即策略：supervisor 排在 terminal 之前是刻意的（见模块 docstring）。
PID_FILES = ("supervisor.pid", "terminal.pid", "backend.pid")
FALLBACK_PORTS = (8501, 8510, 8600)
LOCKFILE = "terminal.lock"


# ---------------------------------------------------------------- 进程工具

def read_pid(path: str) -> int | None:
    """读取 pid 文件；内容非法时返回 None（不抛异常）。"""
    try:
        with open(path, encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    """进程是否存活。psutil 缺失时退回 signal 0 探测。"""
    try:
        import psutil
    except ImportError:
        pass
    else:
        try:
            return psutil.pid_exists(pid)
        except Exception:                                 # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except Exception:                                     # noqa: BLE001
        return False
    return True


def kill_tree(pid: int, timeout: float = 5.0) -> bool:
    """杀掉进程树；成功（进程最终消失）返回 True。"""
    try:
        import psutil
    except ImportError:
        return _kill_tree_fallback(pid)

    try:
        proc = psutil.Process(pid)
        targets = [proc] + proc.children(recursive=True)
    except Exception:                                     # noqa: BLE001
        return False
    for p in targets:
        try:
            p.kill()
        except Exception:                                 # noqa: BLE001
            pass
    try:
        _, alive = psutil.wait_procs(targets, timeout=timeout)
    except Exception:                                     # noqa: BLE001
        alive = []
    return not alive


def _kill_tree_fallback(pid: int) -> bool:
    """无 psutil 时的兜底：Windows 用 taskkill /T /F，POSIX 用 SIGKILL。"""
    if os.name == "nt":
        cmd = ["taskkill", "/PID", str(pid), "/T", "/F"]
    else:
        cmd = ["kill", "-9", str(pid)]
    try:
        rc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15
        ).returncode
    except Exception:                                     # noqa: BLE001
        return False
    return rc == 0


def pids_listening_on(port: int) -> list[int]:
    """列出在该端口 LISTEN 的进程 PID（需 psutil，缺失时返回空列表）。"""
    try:
        import psutil
    except ImportError:
        return []
    found: set[int] = set()
    try:
        conns = psutil.net_connections(kind="tcp")
    except Exception:                                     # noqa: BLE001
        return []
    for c in conns:
        try:
            if c.status == "LISTEN" and c.laddr and c.laddr.port == port and c.pid:
                found.add(c.pid)
        except Exception:                                 # noqa: BLE001
            continue
    return sorted(found)


# ---------------------------------------------------------------- 关停动作

def stop_pidfile(name: str, dry_run: bool = False) -> tuple[bool, str]:
    """按 pid 文件关停。返回 (是否真的杀掉了进程, 人类可读结果)。"""
    path = os.path.join(ROOT, name)
    pid = read_pid(path)
    if pid is None:
        # 陈旧/空 pid 文件：清掉，避免下次启动时误杀无关进程。
        if os.path.exists(path):
            if not dry_run:
                try:
                    os.remove(path)
                except OSError:
                    pass
            return False, f"{name}: 无有效 PID，已清理残留文件"
        return False, f"{name}: 不存在"

    if not pid_alive(pid):
        if not dry_run:
            try:
                os.remove(path)
            except OSError:
                pass
        return False, f"{name}: PID {pid} 已不存在，已清理残留文件"

    if dry_run:
        return True, f"{name}: 将终止 PID {pid}（进程树）"

    ok = kill_tree(pid)
    # 无论成功与否都删掉 pid 文件：进程已死时留着只会误导下次停止。
    try:
        os.remove(path)
    except OSError:
        pass
    if ok:
        return True, f"{name}: 已终止 PID {pid}（进程树）"
    return False, f"{name}: PID {pid} 终止失败，可能需手动处理"


def stop_port(port: int, dry_run: bool = False) -> tuple[bool, str]:
    """兜底：杀掉仍在监听该端口的进程。"""
    pids = pids_listening_on(port)
    if not pids:
        return False, f"端口 {port}: 无监听进程"
    if dry_run:
        return True, f"端口 {port}: 将终止 PID {pids}"
    killed = [p for p in pids if kill_tree(p)]
    if killed:
        return True, f"端口 {port}: 已终止 PID {killed}"
    return False, f"端口 {port}: PID {pids} 终止失败"


def clear_stale_lock(dry_run: bool = False) -> str:
    """清理 start.py 的端口互斥锁（其 _clear_lock 从未被调用，停止时应清掉）。"""
    path = os.path.join(ROOT, LOCKFILE)
    if not os.path.exists(path):
        return f"{LOCKFILE}: 不存在，无需清理"
    try:
        with open(path, encoding="utf-8") as f:
            lock = json.load(f)
    except (OSError, ValueError):
        lock = {}
    pid = lock.get("pid") if isinstance(lock, dict) else None
    alive = isinstance(pid, int) and pid_alive(pid)
    if alive and not dry_run:
        return f"{LOCKFILE}: 锁持有者 PID {pid} 仍在运行，保留"
    if dry_run:
        return f"{LOCKFILE}: 将删除（持有者 PID {pid} 已不在）"
    try:
        os.remove(path)
    except OSError:
        return f"{LOCKFILE}: 删除失败"
    return f"{LOCKFILE}: 已清理（持有者 PID {pid} 已不在）"


def stop_all(dry_run: bool = False, keep_backend: bool = False) -> dict:
    """执行完整关停流程，返回结构化结果（便于测试与上层调用）。"""
    names = [n for n in PID_FILES if not (keep_backend and n == "backend.pid")]
    results = [stop_pidfile(n, dry_run=dry_run) for n in names]

    ports = list(FALLBACK_PORTS)
    if keep_backend:
        ports = [p for p in ports if p != 8600]
    port_results = [stop_port(p, dry_run=dry_run) for p in ports]

    lock_msg = clear_stale_lock(dry_run=dry_run)
    killed_any = any(ok for ok, _ in results) or any(ok for ok, _ in port_results)

    return {
        "killed": killed_any,
        "dry_run": dry_run,
        "pid_files": [{"name": n, "killed": ok, "msg": msg}
                      for n, (ok, msg) in zip(names, results)],
        "ports": [{"port": p, "killed": ok, "msg": msg}
                  for p, (ok, msg) in zip(ports, port_results)],
        "lock": lock_msg,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="停止量化终端 / 守护进程 / 数据后端")
    ap.add_argument("--dry-run", action="store_true",
                    help="只报告将要执行的动作，不做任何关停")
    ap.add_argument("--keep-backend", action="store_true",
                    help="保留真实行情后端（仅停止终端）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args(argv)

    res = stop_all(dry_run=args.dry_run, keep_backend=args.keep_backend)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}停止量化服务" + ("（保留数据后端）" if args.keep_backend else ""))
    for item in res["pid_files"]:
        print(f"  - {prefix}{item['msg']}")
    for item in res["ports"]:
        print(f"  - {prefix}{item['msg']}")
    print(f"  - {prefix}{res['lock']}")
    print(prefix + ("已停止相关服务。" if res["killed"] else "未发现运行中的服务。"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
