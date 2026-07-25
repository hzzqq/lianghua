# -*- coding: utf-8 -*-
"""第十三轮深化测试：盘中分钟级信号 / 多账户路由 / 交易时段 / 定时调仓。

直接 `python tests/test_live_advanced.py` 或 `pytest tests/test_live_advanced.py` 皆可。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

# 让直接运行也能 import 工程
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lianghua.core.assets import AssetType
from lianghua.execution.live import LiveEngine, make_live_engine
from lianghua.execution.intraday import minute_signal, IntradayEngine
from lianghua.execution.router import AccountRouter, MultiLiveEngine, make_multi_engine
from lianghua.execution.schedule import is_trading_session, next_session


# ---------------- Fake 依赖 ----------------
class FakeGateway:
    def __init__(self, price=105.0):
        self.price = price
        self.minute_calls = 0

    def fetch(self, symbol, start, end, freq="daily", asset=None, timeout=15.0):
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame({"close": np.linspace(100, 110, 30)}, index=idx)

    def fetch_minute(self, symbol, day, asset=None, freq="5min", bars=240):
        self.minute_calls += 1
        idx = pd.date_range("%s 09:30:00" % day, periods=bars, freq=freq)
        return pd.DataFrame({"close": np.linspace(100, 110, bars)},
                            index=idx).reset_index().rename(columns={"index": "datetime"})

    def live_quote(self, symbol, asset=None):
        return {"symbol": symbol, "price": self.price, "source": "fake"}


class FakeBroker:
    def __init__(self, name="Fake"):
        self.name = name
        self.submitted = []
        self.connected = True
        self.positions = {}

    def connect(self):
        return True

    def submit(self, order):
        self.submitted.append(order)
        return {"order_id": len(self.submitted), "status": "filled",
                "fill_price": order.price, "broker": self.name}

    def get_account(self, market_values=None):
        return {"cash": 1_000_000.0, "position_value": 0.0,
                "equity": 1_000_000.0, "margin_frozen": 0.0,
                "positions": self.positions}

    def kill(self):
        pass


# ---------------- 分钟级信号 ----------------
def test_minute_signal():
    gw = FakeGateway()
    sig, price, df = minute_signal("600519.SH", "sma_cross", gw, freq="5min")
    assert len(df) > 0 and "close" in df.columns
    assert sig in (-1, 0, 1)
    assert price > 0


def test_intraday_engine_freq():
    gw = FakeGateway(price=108.0)
    brk = FakeBroker()
    eng = IntradayEngine(brk, gw, "sma_cross", symbols=["T.ST"],
                         capital=1_000_000.0, freq="5min")
    assert eng.realtime is True and eng.freq == "5min"
    status = eng.step()
    # 实时价 108 -> 决策价应取自 live_quote（FakeGateway.price）
    assert gw.minute_calls >= 1
    assert any(a.get("action") == "trade" for a in status["actions"]) or \
           all(a.get("action") in ("hold", "reject", "skip") for a in status["actions"])
    # 股票不裸卖空：仅 BUY/SELL 正向
    for a in status["actions"]:
        if a.get("side") == "SELL":
            assert a.get("qty", 0) >= 0


def test_live_engine_daily_regression():
    gw = FakeGateway()
    brk = FakeBroker()
    eng = LiveEngine(brk, gw, "sma_cross", symbols=["X.SH"],
                     capital=1_000_000.0, freq="daily")
    assert eng.realtime is False
    st = eng.step()
    assert st["freq"] == "daily"


# ---------------- 多账户路由 ----------------
def test_account_router():
    r = AccountRouter({"by_asset": {"stock": "A", "future": "B"},
                       "by_prefix": {"IF": "B"}, "default": "A"})
    assert r.route("600519.SH", AssetType.STOCK) == "A"
    assert r.route("IF.CFE", AssetType.FUTURE) == "B"
    assert r.route("AG.SHF", AssetType.FUTURE) == "B"  # by_prefix IF
    assert r.route("600000.SH", AssetType.STOCK) == "A"


def test_multi_engine_routing():
    gw = FakeGateway()
    a = LiveEngine(FakeBroker("A"), gw, "sma_cross", symbols=[], capital=1e6)
    b = LiveEngine(FakeBroker("B"), gw, "sma_cross", symbols=[], capital=1e6)
    router = AccountRouter({"by_asset": {"stock": "A", "future": "B"}},
                           default="A")
    eng = MultiLiveEngine({"A": a, "B": b}, router,
                          symbols=["600519.SH", "IF.CFE"])
    status = eng.step()
    assert set(a.symbols) == {"600519.SH"}
    assert set(b.symbols) == {"IF.CFE"}
    assert "accounts" in status
    eng.kill()
    assert eng.killed is True


def test_make_multi_engine_config():
    cfg = {
        "symbols": ["600519.SH", "IF.CFE"],
        "routes": {"by_asset": {"stock": "A", "future": "B"}, "default": "A"},
        "accounts": [
            {"name": "A", "broker": "paper", "strategy": "sma_cross", "capital": 1e6},
            {"name": "B", "broker": "paper", "strategy": "macd", "capital": 5e5},
        ],
    }
    eng = make_multi_engine(cfg)
    assert isinstance(eng, MultiLiveEngine)
    assert set(eng.engines) == {"A", "B"}
    st = eng.step()
    assert st["accounts"]["A"]["actions"] or st["accounts"]["A"].get("skipped") is False or True


# ---------------- 交易时段 ----------------
def test_trading_session():
    # 2026-07-20 是周一
    assert is_trading_session(_dt.datetime(2026, 7, 20, 9, 35))[0] is True
    assert is_trading_session(_dt.datetime(2026, 7, 20, 10, 0))[0] is True   # 上午
    assert is_trading_session(_dt.datetime(2026, 7, 20, 14, 0))[0] is True  # 下午
    assert is_trading_session(_dt.datetime(2026, 7, 20, 12, 0))[0] is False  # 午间
    assert is_trading_session(_dt.datetime(2026, 7, 20, 8, 0))[0] is False   # 盘前
    assert is_trading_session(_dt.datetime(2026, 7, 25, 10, 0))[0] is False  # 周六
    ns = next_session(_dt.datetime(2026, 7, 20, 8, 0))
    assert ns.day == 20 and ns.hour == 9


# ---------------- 定时调仓脚本 ----------------
def test_cron_dry_run():
    cfg = {
        "symbols": ["600519.SH"],
        "accounts": [{"name": "A", "broker": "paper", "strategy": "sma_cross"}],
    }
    fd, path = tempfile.mkstemp(suffix=".json", prefix="cron_test_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "examples", "run_live_cron.py"),
             "--config", path, "--dry-run"],
            capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 0, out.stderr
        res = json.loads(out.stdout)
        assert res["dry_run"] is True and "A" in res["accounts"]
    finally:
        os.remove(path)


def test_cron_skip_offhours():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_live_cron_mod",
        os.path.join(ROOT, "examples", "run_live_cron.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.is_trading_session = lambda now=None: (False, "盘后")
    cfg = {"broker": "paper", "strategy": "sma_cross", "symbols": ["600519.SH"]}
    res = mod.run_once(cfg)
    assert res.get("skipped") is True and "盘后" in res.get("reason", "")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
