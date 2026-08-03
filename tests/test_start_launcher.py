"""启动器（start.py 与各 .bat/.sh 包装脚本）的回归测试。

启动脚本是用户使用本项目的唯一入口，一旦坏掉整个工具库就不可用，且这类问题
不会被任何业务测试发现。这里钉住几类真实发生过的故障：

1. .bat 带 UTF-8 BOM —— cmd.exe 会报 "'@echo' 不是内部或外部命令"，且 @echo off 失效；
2. 启动脚本写死某台机器的 python 绝对路径 —— 换机器/换用户/Linux 上必然启动失败；
3. 解释器探测只判断"文件存在"而不判断"真的装了 streamlit" —— 报错难以理解；
4. 端口被占用时抛出晦涩的 streamlit 异常；
5. .bat 正文里混入多字节中文 —— 配合 `chcp 65001` 会让 cmd.exe 的批处理解析器
   （按字节偏移回读文件）失步，把注释/变量名的片段当成命令执行，双击直接启动失败。
"""
from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 所有对外承诺的启动入口都必须逐一体检：run.* 是主入口，中文名 .bat 是双击入口，
# start.* 是为兼容旧使用习惯（快捷方式、文档、肌肉记忆）保留的转调别名。
BAT_FILES = ["run.bat", "start.bat", "启动量化终端.bat"]
SH_FILES = ["run.sh", "start.sh"]
BOM = b"\xef\xbb\xbf"


@pytest.mark.parametrize("name", BAT_FILES + SH_FILES + ["start.py"])
def test_declared_launchers_exist(name):
    """清单里的启动入口必须真实存在。

    其余用例遇到缺失文件会 skip，若某个入口被误删，这些 skip 会静默掩盖问题；
    由本用例负责让"入口消失"这件事直接报错。
    """
    assert os.path.exists(os.path.join(ROOT, name)), (
        f"启动入口 {name} 不存在：要么补回文件，要么同步更新本测试与 README"
    )


def _load_start_module():
    path = os.path.join(ROOT, "start.py")
    spec = importlib.util.spec_from_file_location("lianghua_start_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def start_mod():
    return _load_start_module()


@pytest.mark.parametrize("name", BAT_FILES)
def test_bat_has_no_utf8_bom(name):
    """.bat 文件不能有 BOM，否则 cmd.exe 第一行报错且 @echo off 不生效。"""
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 不存在")
    with open(path, "rb") as f:
        head = f.read(3)
    assert head != BOM, f"{name} 带 UTF-8 BOM，会导致 cmd.exe 启动报错"


@pytest.mark.parametrize("name", BAT_FILES + SH_FILES + ["start.py"])
def test_no_machine_specific_python_path(name):
    """启动脚本不得写死某个用户的 python 绝对路径（换机器就废）。"""
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 不存在")
    text = open(path, encoding="utf-8").read().lower()
    for bad in ("c:\\users\\administrator", "c:/users/administrator"):
        assert bad not in text, f"{name} 写死了机器相关路径 {bad}"


@pytest.mark.parametrize("name", BAT_FILES)
def test_bat_body_is_pure_ascii(name):
    """.bat 正文必须是纯 ASCII（文件名可以是中文，那是文件系统层面的事）。

    真实故障：`启动量化终端.bat` 曾被改写成 `chcp 65001` + 中文提示语直接拉起 start.py。
    cmd.exe 执行批处理时会按字节偏移反复回读文件，而 `chcp` 切换代码页后对多字节
    字符的偏移换算会错位，实测输出为
        'UA_PYTHONLIANGHUA_PYTHON"' 不是内部或外部命令
        'thon.exe" set "PY' 不是内部或外部命令
    并以 ERRORLEVEL 9009 退出 —— 用户双击后终端根本起不来。
    结论：中文提示放到 start.py（Python 侧编码可控），.bat 只做纯 ASCII 的引导。
    """
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 不存在")
    raw = open(path, "rb").read()
    bad = [(i, b) for i, b in enumerate(raw) if b > 0x7F]
    assert not bad, (
        f"{name} 正文含非 ASCII 字节（首个位于偏移 {bad[0][0]}）："
        "与 chcp 65001 组合会导致 cmd.exe 解析错位、启动失败"
    )


@pytest.mark.skipif(os.name != "nt", reason="仅 Windows 下 cmd.exe 可执行 .bat")
@pytest.mark.parametrize("name", BAT_FILES)
def test_bat_actually_runs(name):
    """真的用 cmd.exe 跑一遍 .bat，确认它能被解析并把参数透传给 start.py。

    静态检查挡不住所有批处理坑（块语句、转义、编码），这里用 `--help` 做低成本冒烟：
    argparse 在解释器探测之前就会打印用法并以 0 退出，几百毫秒即可完成。
    """
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 不存在")
    r = subprocess.run(
        ["cmd.exe", "/c", name, "--help"],
        cwd=ROOT, stdin=subprocess.DEVNULL,
        capture_output=True, timeout=300,
    )
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    assert r.returncode == 0, f"{name} 执行失败 (rc={r.returncode})：\n{out}"
    assert "usage: start.py" in out, f"{name} 未把参数正确透传给 start.py：\n{out}"


@pytest.mark.skipif(
    os.name != "nt" or not os.environ.get("LIANGHUA_E2E"),
    reason="端到端拉起 Streamlit 耗时较长，需设 LIANGHUA_E2E=1 显式开启",
)
def test_double_click_entry_serves_healthy_app():
    """端到端：走真实双击入口把终端拉起来，确认 Streamlit 健康检查返回 ok。

    `--help` 冒烟只证明 .bat 能被解析，证明不了"双击之后页面真的出得来"。
    这里补上最后一段：入口 -> run.bat -> start.py -> streamlit，然后打
    `/_stcore/health`。默认不跑（约需十几秒且要占一个端口），CI/本地排障时
    用 `LIANGHUA_E2E=1 pytest tests/test_start_launcher.py` 开启。

    只对本用例自己 spawn 出来的进程组发 CTRL_BREAK，不触碰任何其它进程。
    """
    import signal
    import urllib.request

    port = int(os.environ.get("LIANGHUA_E2E_PORT", "8599"))
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            pytest.skip(f"端口 {port} 已被占用")

    env = {**os.environ, "LIANGHUA_PYTHON": sys.executable}
    proc = subprocess.Popen(
        ["cmd.exe", "/c", "启动量化终端.bat", "--no-browser", "--port", str(port)],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    body = ""
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            assert proc.poll() is None, "启动入口在服务就绪前就退出了"
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/_stcore/health", timeout=2
                ) as r:
                    body = r.read().decode("utf-8", "replace").strip()
                if body == "ok":
                    break
            except OSError:
                time.sleep(1.5)
        assert body == "ok", f"健康检查未通过（收到 {body!r}）"
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover - 兜底
                proc.kill()
                proc.wait(timeout=15)


@pytest.mark.parametrize("name", SH_FILES)
def test_sh_scripts_are_portable(name):
    """.sh 声称支持 Linux/macOS，就不能依赖 Windows 专有的 .exe 绝对路径。"""
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 不存在")
    text = open(path, encoding="utf-8").read()
    assert "python.exe" not in text or "$" in text, f"{name} 在非 Windows 上无法运行"


@pytest.mark.parametrize("name", SH_FILES)
def test_sh_scripts_have_lf_endings(name):
    """.sh 必须是 LF 换行，CRLF 会让 shebang 变成 `#!/usr/bin/env bash\\r`。

    真实故障模式：Git for Windows 默认 `core.autocrlf=true`，克隆时把 .sh 一起转成
    CRLF；在 Git Bash / WSL / Linux 上执行就报
        bash\\r: bad interpreter: No such file or directory
    —— README 承诺的 "Git Bash / Linux 可运行" 直接不成立。防线是 .gitattributes
    里的 `*.sh text eol=lf`（见 test_gitattributes_pins_launcher_eol）。
    """
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 不存在")
    raw = open(path, "rb").read()
    assert b"\r" not in raw, f"{name} 含 CR 字节，在 Git Bash/Linux 上 shebang 会失效"


@pytest.mark.parametrize("name", SH_FILES)
def test_sh_scripts_are_executable_in_git_index(name):
    """.sh 在 git 索引里必须是 100755，否则 Linux 上 clone 出来没有 +x 位。

    Windows 侧 `core.filemode=false`，本地文件系统的权限位不会被 git 采信，所以
    "本机 ls -l 看着是 rwxr-xr-x" 完全不能说明问题 —— 只有索引里的 mode 才会跟着
    仓库走。mode 是 100644 时，Linux 用户 clone 后执行 `./run.sh` 会得到
    "Permission denied"，只能退而求其次 `bash run.sh`。
    修复方式：`git update-index --chmod=+x <file>`。
    """
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 不存在")
    r = subprocess.run(
        ["git", "ls-files", "-s", "--", name],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0 or not r.stdout.strip():
        pytest.skip("git 不可用或文件未纳入版本控制")
    mode = r.stdout.split()[0]
    assert mode == "100755", (
        f"{name} 在 git 索引里的 mode 是 {mode}，Linux 上 clone 后不可直接执行；"
        f"请运行 git update-index --chmod=+x {name}"
    )


def test_gitattributes_pins_launcher_eol():
    """必须有 .gitattributes 锁死启动脚本换行符，不能听凭各人的 autocrlf 设置。

    没有它时，同一个仓库在 `core.autocrlf=true`（Git for Windows 默认）和
    `false` 两台机器上 checkout 出来的 .sh/.bat 换行符不同：前者的 .sh 会被
    转成 CRLF 而无法运行。这是"在我机器上是好的"类跨平台故障的根源。
    """
    path = os.path.join(ROOT, ".gitattributes")
    assert os.path.exists(path), "缺少 .gitattributes，跨平台换行符无保障"
    text = open(path, encoding="utf-8").read()
    assert "*.sh text eol=lf" in text, ".gitattributes 未把 *.sh 锁成 LF"
    assert "*.bat text eol=crlf" in text, ".gitattributes 未把 *.bat 锁成 CRLF"


def test_no_dangling_launcher_references():
    """README 和各启动脚本不得引用已不存在的启动文件。

    真实故障：一次"入口整合"把 `start.bat` / `start.sh` 删掉了，而 README 里仍有
    "双击 start.bat / 运行 start.sh" 的说明 —— 用户照做就是文件不存在。文档与磁盘
    上的真实入口必须同步，本用例负责钉死这一点。
    """
    import re

    sources = ["README.md", "start.py", *BAT_FILES, *SH_FILES]
    # 先抹掉脚本里的路径前缀（cmd 的 %~dp0、shell 的 ./ 等），只留文件名
    prefix = re.compile(r"%~dp0|\$\{?BASEDIR\}?/|\./")
    pattern = re.compile(r"([\w\u4e00-\u9fff\-]+\.(?:bat|sh))\b")
    dangling: list[str] = []
    for src in sources:
        src_path = os.path.join(ROOT, src)
        if not os.path.exists(src_path):
            continue
        text = prefix.sub(" ", open(src_path, encoding="utf-8").read())
        for ref in set(pattern.findall(text)):
            if not os.path.exists(os.path.join(ROOT, ref)):
                dangling.append(f"{src} -> {ref}")
    assert not dangling, "引用了不存在的启动脚本：" + "; ".join(sorted(dangling))


def test_candidate_pythons_are_real_and_deduped(start_mod):
    cands = start_mod._candidate_pythons()
    assert all(os.path.isfile(c) for c in cands), "候选解释器必须真实存在"
    keys = [os.path.normcase(os.path.abspath(c)) for c in cands]
    assert len(set(keys)) == len(keys), "候选解释器不应重复"


def test_has_streamlit_is_a_real_probe(start_mod):
    """必须真的去 import，而不是只看文件在不在。"""
    assert start_mod._has_streamlit(os.path.join(ROOT, "no_such_python.exe")) is False
    expected = importlib.util.find_spec("streamlit") is not None
    assert start_mod._has_streamlit(sys.executable) is expected


def test_resolve_python_returns_interpreter_with_streamlit(start_mod):
    py = start_mod._resolve_python()
    if py is None:
        # 环境里确实没有装 streamlit 的解释器：此时必须返回 None 让 main() 给出友好提示
        assert importlib.util.find_spec("streamlit") is None
        return
    assert os.path.isfile(py) or py == sys.executable
    assert start_mod._has_streamlit(py), "解析出的解释器必须真的能 import streamlit"


def test_port_in_use_detection(start_mod):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    port = srv.getsockname()[1]
    try:
        assert start_mod._port_in_use("127.0.0.1", port) is True
        # 监听 0.0.0.0 时应回落到 loopback 探测，而不是去连 0.0.0.0
        assert start_mod._port_in_use("0.0.0.0", port) is True
    finally:
        srv.close()
    assert start_mod._port_in_use("127.0.0.1", port) is False


def test_default_host_is_loopback(start_mod, monkeypatch):
    """默认只监听本机：量化终端没有鉴权，默认对局域网开放是安全隐患。"""
    monkeypatch.setattr(sys, "argv", ["start.py"])
    import argparse

    parsed = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *a, **kw):
        ns = real_parse(self, *a, **kw)
        parsed.update(vars(ns))
        raise SystemExit(0)  # 不真的去拉起 streamlit

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    with pytest.raises(SystemExit):
        start_mod.main()
    assert parsed["host"] == "127.0.0.1"
    assert parsed["port"] == 8501
