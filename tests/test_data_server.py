"""真实行情后端（backend/data_server.py）鉴权与只读模式回归。

R18 安全加固（进度文档 Backlog：后端 CORS `*` 且无鉴权，仅限本机使用前提）：
- 默认监听 127.0.0.1（原 0.0.0.0 对局域网开放而无鉴权）；
- ``--token`` / 环境变量 ``LIANGHUA_BACKEND_TOKEN``：除 /api/health 心跳外，
  全部端点要求请求头 ``X-Api-Token`` 或查询参数 ``?token=``（SSE 走查询参数）；
- ``--read-only``：禁用 /api/refresh（唯一会写缓存的端点），其余端点天然只读。

全程用桩网关（Stub Gateway），不触真实网络、不写真实缓存库。
"""
import importlib.util
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_data_server():
    path = os.path.join(ROOT, "backend", "data_server.py")
    spec = importlib.util.spec_from_file_location("lianghua_data_server_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubGW:
    """最小网关桩：只暴露后端用到的面。"""

    def __init__(self):
        self.last_source = "akshare"
        self.last_was_demo = False
        self._warns: list[str] = []

    def last_warnings(self):
        return list(self._warns)

    def fetch(self, symbol, start, end, asset=None, timeout=10.0, force_refresh=False):
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        close = 100.0 * np.cumprod(1.0 + np.full(len(dates), 0.001))
        return pd.DataFrame({"date": dates, "open": close, "high": close * 1.01,
                             "low": close * 0.99, "close": close, "volume": 1e6,
                             "source": "akshare"})

    def live_quote(self, symbol, asset=None):
        return {"symbol": symbol, "price": 100.0, "source": "akshare"}

    def list_cached(self, asset=None):
        return {"600519.SH": {"min": "2024-01-01", "max": "2024-01-10",
                              "rows": 10, "demo_rows": 0, "source": "akshare"}}


def _make_server(ds, token="", read_only=False):
    # 绕过 Backend.__init__：避免创建真实 DataGateway（会连真实缓存库/网络）
    b = ds.Backend.__new__(ds.Backend)
    b.gw = _StubGW()
    b.real_timeout = 5.0
    b.watchlist = []
    b.live_cache = {}
    b._preclose_cache = {}
    b.live_lock = threading.Lock()
    b._stop = threading.Event()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ds.Handler)
    srv.backend = b
    srv.auth_token = token
    srv.read_only = read_only
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture(scope="module")
def ds():
    return _load_data_server()


@pytest.fixture
def http():
    """禁代理的 GET 帮手：返回 (status, json)。沙箱强制代理会劫持请求。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _get(url, headers=None):
        req = urllib.request.Request(url, headers=headers or {})
        try:
            with opener.open(req, timeout=5) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8")
            try:
                return e.code, json.loads(raw or "{}")
            except ValueError:
                return e.code, {"raw": raw}

    return _get


@pytest.fixture
def anon_srv(ds):
    srv = _make_server(ds)
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def auth_srv(ds):
    srv = _make_server(ds, token="s3cret")
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def ro_srv(ds):
    srv = _make_server(ds, read_only=True)
    yield srv
    srv.shutdown()
    srv.server_close()


def _base(srv) -> str:
    return f"http://127.0.0.1:{srv.server_address[1]}"


_HISTORY = "/api/history?symbol=600519.SH&start=2024-01-01&end=2024-01-10"


# ---------- 无令牌（默认，行为不退化） ----------

def test_no_token_serves_history(anon_srv, http):
    code, body = http(f"{_base(anon_srv)}{_HISTORY}")
    assert code == 200
    assert body["rows"] > 0 and body["source"] == "akshare"


def test_no_token_refresh_still_works(anon_srv, http):
    code, body = http(f"{_base(anon_srv)}/api/refresh?symbol=600519.SH"
                      "&start=2024-01-01&end=2024-01-10")
    assert code == 200 and body["rows"] > 0


def test_cache_endpoint_reports_demo_rows(anon_srv, http):
    """/api/cache 透传 list_cached：demo_rows 分离标记必须可达终端。"""
    code, body = http(f"{_base(anon_srv)}/api/cache")
    assert code == 200
    assert "600519.SH" in body
    assert "demo_rows" in body["600519.SH"]


# ---------- 令牌鉴权 ----------

def test_health_never_requires_token(auth_srv, http):
    """心跳保持无鉴权：终端 2~5s 高频轮询它判断后端存活。"""
    code, body = http(f"{_base(auth_srv)}/api/health")
    assert code == 200 and body["status"] == "up"


def test_token_missing_rejected(auth_srv, http):
    code, body = http(f"{_base(auth_srv)}{_HISTORY}")
    assert code == 403 and "令牌" in body.get("error", "")


def test_token_wrong_rejected(auth_srv, http):
    code, _ = http(f"{_base(auth_srv)}{_HISTORY}&token=wrong")
    assert code == 403


def test_token_via_query_param(auth_srv, http):
    """SSE/EventSource 无法设请求头，必须支持 ?token= 查询参数。"""
    code, body = http(f"{_base(auth_srv)}{_HISTORY}&token=s3cret")
    assert code == 200 and body["rows"] > 0


def test_token_via_header(auth_srv, http):
    code, body = http(f"{_base(auth_srv)}{_HISTORY}", headers={"X-Api-Token": "s3cret"})
    assert code == 200 and body["rows"] > 0


def test_token_covers_index_page(auth_srv, http):
    """除 /api/health 外全部要求令牌（含 / 说明页，防端点枚举）。"""
    code, _ = http(f"{_base(auth_srv)}/")
    assert code == 403


# ---------- 只读模式 ----------

def test_read_only_blocks_refresh(ro_srv, http):
    code, body = http(f"{_base(ro_srv)}/api/refresh?symbol=600519.SH"
                      "&start=2024-01-01&end=2024-01-10")
    assert code == 403 and "只读" in body.get("error", "")


def test_read_only_keeps_read_routes(ro_srv, http):
    code, _ = http(f"{_base(ro_srv)}{_HISTORY}")
    assert code == 200
    code2, _ = http(f"{_base(ro_srv)}/api/health")
    assert code2 == 200


# ---------- 启动参数默认值 ----------

def _capture_parse_args(ds, monkeypatch, argv, env=None):
    import argparse
    captured = {}
    real = argparse.ArgumentParser.parse_args

    def spy(self, *a, **kw):
        ns = real(self, *a, **kw)
        captured.update(vars(ns))
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    monkeypatch.setattr(sys, "argv", argv)
    if env is not None:
        monkeypatch.setenv("LIANGHUA_BACKEND_TOKEN", env)
    with pytest.raises(SystemExit):
        ds.main()
    return captured


def test_default_host_is_loopback_and_auth_off(ds, monkeypatch):
    """默认仅本机监听 + 不鉴权 + 非只读（行为不退化的安全底线）。"""
    cap = _capture_parse_args(ds, monkeypatch, ["data_server.py"])
    assert cap["host"] == "127.0.0.1"
    assert cap["token"] == ""
    assert cap["read_only"] is False


def test_token_default_from_env(ds, monkeypatch):
    cap = _capture_parse_args(ds, monkeypatch, ["data_server.py"], env="env-secret")
    assert cap["token"] == "env-secret"
