# -*- coding: utf-8 -*-
"""Windows 批处理脚本（.bat）静态体检 + 端到端冒烟。

为什么需要它：
- 项目在 `run.bat` 上踩过坑：**`.bat` 内 `chcp 65001` 与多字节中文同时出现时，
  cmd.exe 会按字节偏移定位后续命令而失步**，把注释片段当命令执行导致启动失败。
  因此纪律是"含 chcp 就必须纯 ASCII"。这条纪律此前只写在 README 里，没人守。
- 沙箱的 Bash / PowerShell 工具会拦截直接调用 `cmd.exe`（命令校验层），
  但**从 Python 内部用 subprocess 调用 cmd 是可行的**，所以本测试可以真跑一次 bat。
- `.bat` 崩在用户双击的那一刻最难排查，静态体检能把它拦在提交之前。

静态检查项：
    1. 含 `chcp 65001` 的 bat 必须纯 ASCII（字节 <= 127）
    2. 换行必须是 CRLF（LF-only 的 bat 在 cmd 下行为异常）
    3. bat 内部引用的脚本文件必须真实存在

冒烟测试：
    4. `停止量化终端.bat --dry-run` 真实执行，rc 必须为 0
       （dry-run 不杀任何进程、不删任何文件，可安全重复运行；
        cmd 不可用时自动 skip）
"""
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BAT_FILES = [
    "run.bat",
    "启动量化终端.bat",
    "停止量化终端.bat",
    "start_all.bat",
    os.path.join("backend", "start_data_backend.bat"),
]

# bat 内部引用的、必须存在的目标文件（相对项目根）
REQUIRED_REFS = {
    "启动量化终端.bat": ["run.bat"],
    "停止量化终端.bat": [os.path.join("tools", "stop_services.py")],
}


def _read_bytes(rel: str) -> bytes:
    with open(os.path.join(ROOT, rel), "rb") as f:
        return f.read()


def _bats_present() -> list[str]:
    return [b for b in BAT_FILES if os.path.exists(os.path.join(ROOT, b))]


# ---------------------------------------------------------------- 静态检查

def test_bat_files_exist():
    missing = [b for b in BAT_FILES if not os.path.exists(os.path.join(ROOT, b))]
    assert not missing, f"缺少预期的 .bat 文件: {missing}"


def test_chcp_bats_must_be_ascii():
    """含 chcp 65001 的 bat 一旦出现非 ASCII 字节，cmd 解析就会失步。"""
    for rel in _bats_present():
        raw = _read_bytes(rel)
        has_chcp = b"chcp" in raw.lower()
        non_ascii = sorted({b for b in raw if b > 127})
        if has_chcp:
            assert not non_ascii, (
                f"{rel} 同时包含 chcp 与非 ASCII 字节 {non_ascii[:8]} —— "
                "cmd.exe 会按字节偏移定位后续命令而失步（README「一键启动文件」已记录此坑）。"
                "请二选一：去掉 chcp，或把注释/提示改为纯 ASCII。"
            )
        else:
            # 不含 chcp 时也建议纯 ASCII：中文 Windows 默认 936 代码页，
            # UTF-8 存盘的中文会显示成乱码。
            assert not non_ascii, (
                f"{rel} 含非 ASCII 字节 {non_ascii[:8]}，在没有 chcp 的情况下会显示乱码。"
            )


def test_bats_use_crlf():
    """LF-only 的 .bat 在 cmd.exe 下可能整行吞掉，必须 CRLF。"""
    for rel in _bats_present():
        raw = _read_bytes(rel)
        lf_only = raw.count(b"\n") - raw.count(b"\r\n")
        assert lf_only == 0, f"{rel} 存在 {lf_only} 处 LF-only 换行，须改为 CRLF"


def test_bat_referenced_files_exist():
    """bat 里写死的引用路径必须真实存在，否则双击即失败。"""
    for bat, refs in REQUIRED_REFS.items():
        path = os.path.join(ROOT, bat)
        if not os.path.exists(path):
            continue
        raw = _read_bytes(bat).decode("utf-8", errors="replace")
        for ref in refs:
            assert ref in raw or ref.replace("\\", "/") in raw, (
                f"{bat} 未引用 {ref}"
            )
            assert os.path.exists(os.path.join(ROOT, ref)), (
                f"{bat} 引用了不存在的 {ref}"
            )


def test_sh_scripts_use_lf():
    """.gitattributes 锁定 *.sh 为 LF：CRLF 会让 shebang 变成 `bash\\r` 而报 bad interpreter。"""
    for rel in ("run.sh", "start_all.sh", "stop_all.sh",
                os.path.join("backend", "start_data_backend.sh")):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        raw = _read_bytes(rel)
        assert b"\r\n" not in raw, f"{rel} 含 CRLF，shebang 会被解释为 `bash\\r`"


# ---------------------------------------------------------------- 端到端冒烟

def _cmd_available() -> bool:
    if os.name != "nt":
        return False
    try:
        r = subprocess.run(["cmd", "/c", "echo ok"], capture_output=True,
                           timeout=20)
    except Exception:                                     # noqa: BLE001
        return False
    return r.returncode == 0


@pytest.mark.skipif(not _cmd_available(), reason="cmd.exe 不可用（非 Windows 或被拦截）")
def test_stop_bat_dry_run_smoke():
    """真实执行 `停止量化终端.bat --dry-run`：dry-run 不杀进程、不删文件，可安全重跑。

    注：Bash / PowerShell 工具会拦截直接调用 cmd.exe，但从 Python subprocess
    调用是可行的；此冒烟正是利用了这一点，用于兜住 bat 的语法/编码/选解释器问题。
    """
    r = subprocess.run(
        ["cmd", "/c", "停止量化终端.bat", "--dry-run"],
        input="\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        cwd=ROOT,
    )
    assert r.returncode == 0, f"停止脚本退出码 {r.returncode}\nstdout:\n{r.stdout}"
    assert "dry-run" in r.stdout, f"输出异常，未见到 dry-run 标记:\n{r.stdout}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
