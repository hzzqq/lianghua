"""启动器（start.py 与各 .bat/.sh 包装脚本）的回归测试。

启动脚本是用户使用本项目的唯一入口，一旦坏掉整个工具库就不可用，且这类问题
不会被任何业务测试发现。这里钉住几类真实发生过的故障：

1. .bat 带 UTF-8 BOM —— cmd.exe 会报 "'@echo' 不是内部或外部命令"，且 @echo off 失效；
2. 启动脚本写死某台机器的 python 绝对路径 —— 换机器/换用户/Linux 上必然启动失败；
3. 解释器探测只判断"文件存在"而不判断"真的装了 streamlit" —— 报错难以理解；
4. 端口被占用时抛出晦涩的 streamlit 异常。
"""
from __future__ import annotations

import importlib.util
import os
import socket
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAT_FILES = ["run.bat", "start.bat", "启动量化终端.bat"]
SH_FILES = ["run.sh", "start.sh"]
BOM = b"\xef\xbb\xbf"


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


@pytest.mark.parametrize("name", SH_FILES)
def test_sh_scripts_are_portable(name):
    """.sh 声称支持 Linux/macOS，就不能依赖 Windows 专有的 .exe 绝对路径。"""
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 不存在")
    text = open(path, encoding="utf-8").read()
    assert "python.exe" not in text or "$" in text, f"{name} 在非 Windows 上无法运行"


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
