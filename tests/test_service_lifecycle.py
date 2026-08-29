# -*- coding: utf-8 -*-
"""服务生命周期测试：supervisor / 终端 / 数据后端的启动关停顺序。

背景（本轮修掉的三个真实 bug）：
- supervisor 会在 streamlit 被杀后 2s 自愈重启，因此**必须先杀 supervisor**，
  否则终端永远停不掉。这个顺序是硬约束，用测试锁死防止后人改坏。
- 原 `停止量化终端.bat` 在 `for` 循环内用 `%PID%` 读取刚刚 `set /p` 的值，
  因缺少 `EnableDelayedExpansion` 而展开为空 → taskkill 静默失效。
  逻辑已迁入 `tools/stop_services.py`（可单测、跨平台、绕开 cmd 编码坑）。
- `start.py` 的 `_clear_lock()` 从未被调用 → 停止后 `terminal.lock` 残留。

本测试全部离线：除「真实杀进程」一项使用自身派生的休眠子进程外，
不启动 streamlit、不依赖网络。
"""
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import stop_services as ss  # noqa: E402


# ------------------------------------------------------------------ 顺序不变量

def test_supervisor_must_be_killed_before_terminal():
    """核心不变量：supervisor 排在 terminal 之前，否则会被自愈重启抵消。"""
    assert "supervisor.pid" in ss.PID_FILES, "supervisor.pid 必须参与关停"
    assert "terminal.pid" in ss.PID_FILES
    assert ss.PID_FILES.index("supervisor.pid") < ss.PID_FILES.index("terminal.pid"), (
        "supervisor 必须先于 terminal 停止：否则它会在 2s 内把 streamlit 重新拉起"
    )


def test_backend_pid_participates():
    assert "backend.pid" in ss.PID_FILES


def test_fallback_ports_cover_all_services():
    """默认端口 8501 / 常驻端口 8510 / 后端 8600 都要兜底覆盖。"""
    assert 8501 in ss.FALLBACK_PORTS
    assert 8510 in ss.FALLBACK_PORTS
    assert 8600 in ss.FALLBACK_PORTS


# ------------------------------------------------------------------ pid 读取

def test_read_pid_valid(tmp_path):
    p = tmp_path / "x.pid"
    p.write_text("12345\n", encoding="utf-8")
    assert ss.read_pid(str(p)) == 12345


def test_read_pid_garbage_returns_none(tmp_path):
    p = tmp_path / "x.pid"
    p.write_text("not-a-pid", encoding="utf-8")
    assert ss.read_pid(str(p)) is None


def test_read_pid_missing_returns_none(tmp_path):
    assert ss.read_pid(str(tmp_path / "nope.pid")) is None


def test_pid_alive_current_process():
    assert ss.pid_alive(os.getpid()) is True


def test_pid_alive_nonexistent():
    # 一个几乎不可能存在的 PID
    assert ss.pid_alive(999999) is False


# ------------------------------------------------------------------ 关停动作

def test_stop_pidfile_cleans_stale_file(tmp_path, monkeypatch):
    """PID 已不存在时：清理残留文件，且不报错。"""
    monkeypatch.setattr(ss, "ROOT", str(tmp_path))
    (tmp_path / "terminal.pid").write_text("999999", encoding="utf-8")
    killed, msg = ss.stop_pidfile("terminal.pid")
    assert killed is False
    assert "已不存在" in msg
    assert not (tmp_path / "terminal.pid").exists(), "陈旧 pid 文件应被清理"


def test_stop_pidfile_cleans_garbage_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "ROOT", str(tmp_path))
    (tmp_path / "backend.pid").write_text("garbage", encoding="utf-8")
    killed, msg = ss.stop_pidfile("backend.pid")
    assert killed is False
    assert "无有效 PID" in msg
    assert not (tmp_path / "backend.pid").exists()


def test_stop_pidfile_dry_run_is_inert(tmp_path, monkeypatch):
    """dry-run 不得杀进程、不得删文件。"""
    monkeypatch.setattr(ss, "ROOT", str(tmp_path))
    (tmp_path / "supervisor.pid").write_text(str(os.getpid()), encoding="utf-8")
    killed, msg = ss.stop_pidfile("supervisor.pid", dry_run=True)
    assert killed is True
    assert "将终止" in msg
    assert (tmp_path / "supervisor.pid").exists(), "dry-run 不应删除 pid 文件"
    assert ss.pid_alive(os.getpid()), "dry-run 不应终止任何进程"


def test_stop_pidfile_kills_real_process(tmp_path, monkeypatch):
    """真实关停：派生的休眠子进程应被终止（整棵树）。"""
    monkeypatch.setattr(ss, "ROOT", str(tmp_path))
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # 等子进程真正进入运行态
        for _ in range(50):
            if ss.pid_alive(proc.pid):
                break
            time.sleep(0.1)
        (tmp_path / "supervisor.pid").write_text(str(proc.pid), encoding="utf-8")
        killed, msg = ss.stop_pidfile("supervisor.pid")
        assert killed is True, msg
        assert "已终止" in msg
        for _ in range(50):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        assert proc.poll() is not None, "子进程应已被终止"
        assert not (tmp_path / "supervisor.pid").exists()
    finally:
        if proc.poll() is None:
            proc.kill()


# ------------------------------------------------------------------ 锁清理

def test_clear_stale_lock_removes_dead_owner(tmp_path, monkeypatch):
    """持有者已死的 terminal.lock 必须被清理（start.py 从不自己清）。"""
    monkeypatch.setattr(ss, "ROOT", str(tmp_path))
    import json
    (tmp_path / "terminal.lock").write_text(
        json.dumps({"port": 8510, "pid": 999999, "app": "lianghua"}),
        encoding="utf-8",
    )
    msg = ss.clear_stale_lock()
    assert "已清理" in msg
    assert not (tmp_path / "terminal.lock").exists()


def test_clear_stale_lock_keeps_live_owner(tmp_path, monkeypatch):
    """持有者还活着时不得删锁，否则会破坏 start.py 的端口互斥语义。"""
    monkeypatch.setattr(ss, "ROOT", str(tmp_path))
    import json
    (tmp_path / "terminal.lock").write_text(
        json.dumps({"port": 8510, "pid": os.getpid(), "app": "lianghua"}),
        encoding="utf-8",
    )
    msg = ss.clear_stale_lock()
    assert "保留" in msg
    assert (tmp_path / "terminal.lock").exists()


def test_clear_stale_lock_missing_is_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "ROOT", str(tmp_path))
    assert "无需清理" in ss.clear_stale_lock()


# ------------------------------------------------------------------ 整体流程

def test_stop_all_dry_run_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "ROOT", str(tmp_path))
    res = ss.stop_all(dry_run=True)
    assert res["dry_run"] is True
    assert res["killed"] is False, "空目录下 dry-run 不应有可杀目标"
    # 三个 pid 文件都要被检查到
    assert [x["name"] for x in res["pid_files"]] == list(ss.PID_FILES)
    assert [x["port"] for x in res["ports"]] == list(ss.FALLBACK_PORTS)
    assert "lock" in res


def test_stop_all_keep_backend_skips_backend(tmp_path, monkeypatch):
    """--keep-backend 时应跳过 backend.pid 与 8600 端口。"""
    monkeypatch.setattr(ss, "ROOT", str(tmp_path))
    res = ss.stop_all(dry_run=True, keep_backend=True)
    names = [x["name"] for x in res["pid_files"]]
    assert "backend.pid" not in names
    assert "supervisor.pid" in names and "terminal.pid" in names
    assert 8600 not in [x["port"] for x in res["ports"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
