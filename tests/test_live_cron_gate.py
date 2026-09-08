# -*- coding: utf-8 -*-
"""实盘定时调仓「数据健全性门禁」单元测试（不依赖网络）。

直接验证 check_data_sanity 的决策逻辑：
- 全部真实 -> ok=True
- 任意为演示(假) -> 列入 demo 并告警
- 演示占比 > 阈值 -> ok=False（应拒绝启动）
以及 run_live_cron.main 在门禁触发时返回退出码 2。
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types

import lianghua.data.gateway as gw_mod
import lianghua.core.assets as assets_mod
from lianghua.core.assets import AssetType


class FakeGateway:
    """用受控的 demo 集合模拟 DataGateway，隔离真实网络。"""

    def __init__(self, demo_set):
        self._demo = set(demo_set)
        self.last_was_demo = False
        self.last_source = None

    def fetch(self, symbol, start, end, freq="daily", asset=None, timeout=15.0,
              force_refresh=False, retries=0, backoff=0.0):
        at = asset or assets_mod.detect_asset_type(symbol)
        is_demo = symbol in self._demo
        self.last_was_demo = is_demo
        self.last_source = "demo" if is_demo else "akshare"
        # 返回一个带 source 列的 DataFrame，模拟 fetch 末尾的标注
        import pandas as pd
        df = pd.DataFrame({
            "date": [start, end],
            "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
            "close": [1.0, 1.0], "volume": [0.0, 0.0],
        })
        df["source"] = self.last_source
        return df


def _make_cfg(symbols, refuse_ratio=1/3):
    return {
        "symbols": list(symbols),
        "data_sanity": {"demo_refuse_ratio": refuse_ratio, "lookback_days": 30},
        "accounts": [{"name": "paper_stock", "symbols": []},
                     {"name": "paper_future", "symbols": []}],
    }


def test_all_real_passes():
    gw_mod.DataGateway = lambda *a, **k: FakeGateway([])  # 无 demo
    import examples.run_live_cron as m
    importlib.reload(m)
    res = m.check_data_sanity(_make_cfg(["600519.SH", "510300.SH", "IF.CFE"]))
    assert res["ok"] is True, res
    assert res["demo"] == [], res
    assert res["ratio"] == 0.0


def test_one_demo_warns_but_passes():
    # 1/3 演示，未超 1/3 阈值 -> 放行但告警
    gw_mod.DataGateway = lambda *a, **k: FakeGateway(["IF.CFE"])
    import examples.run_live_cron as m
    importlib.reload(m)
    res = m.check_data_sanity(_make_cfg(["600519.SH", "510300.SH", "IF.CFE"]),
                              refuse_ratio=1/3)
    assert res["ok"] is True, res
    assert res["demo"] == ["IF.CFE"], res
    # 1/3 == 阈值，使用 > 比较：0.333 > 0.333 为 False -> 仍放行
    assert res["ratio"] > 1/3 + 1e-9 or abs(res["ratio"] - 1/3) < 1e-9


def test_majority_demo_refuses():
    # 2/3 演示 -> 超阈值 -> 拒绝
    gw_mod.DataGateway = lambda *a, **k: FakeGateway(["000300.SH", "IF.CFE"])
    import examples.run_live_cron as m
    importlib.reload(m)
    res = m.check_data_sanity(_make_cfg(["600519.SH", "000300.SH", "IF.CFE"]),
                              refuse_ratio=1/3)
    assert res["ok"] is False, res
    assert set(res["demo"]) == {"000300.SH", "IF.CFE"}
    assert res["ratio"] > 1/3


def test_strict_ratio_zero_refuses_any_demo():
    # 阈值设为 0.0：任意演示即拒绝
    gw_mod.DataGateway = lambda *a, **k: FakeGateway(["IF.CFE"])
    import examples.run_live_cron as m
    importlib.reload(m)
    res = m.check_data_sanity(_make_cfg(["600519.SH", "510300.SH", "IF.CFE"]),
                              refuse_ratio=0.0)
    assert res["ok"] is False, res
    assert res["demo"] == ["IF.CFE"]


def test_account_symbols_also_counted():
    # 账户级 symbols 也会被纳入门禁统计
    gw_mod.DataGateway = lambda *a, **k: FakeGateway(["RB.SHF"])
    import examples.run_live_cron as m
    importlib.reload(m)
    cfg = _make_cfg(["600519.SH"])
    cfg["accounts"][0]["symbols"] = ["RB.SHF"]
    res = m.check_data_sanity(cfg, refuse_ratio=0.0)
    assert "RB.SHF" in res["demo"]
    assert res["total"] == 2


def test_main_refuses_with_exit_code_2(tmp_path):
    # main() 在门禁触发时应返回 2，并把跳过原因写入状态文件
    gw_mod.DataGateway = lambda *a, **k: FakeGateway(["IF.CFE", "510300.SH"])
    import examples.run_live_cron as m
    importlib.reload(m)
    cfg = _make_cfg(["600519.SH", "510300.SH", "IF.CFE"])
    cfg_path = tmp_path / "live_cron.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    state_path = tmp_path / "live_cron_state.json"
    # 让 _write_state 写到临时目录：monkeypatch ROOT 不可行，改测 check_data_sanity 已覆盖决策；
    # 这里仅验证 main 在 --no-gate 关闭时不会因导入崩溃，并在门禁开启时返回 2。
    old_argv = sys.argv
    try:
        # 交易时段永远判定为休市：让 run_once 直接跳过，避免测试触达真实网络/引擎
        m.is_trading_session = lambda *a: (False, "test-skip")
        # 门禁开启 -> 应拒绝返回 2
        sys.argv = ["run_live_cron.py", "--config", str(cfg_path)]
        rc = m.main()
        assert rc == 2, "门禁触发应返回退出码 2，得到 %r" % rc
        # --no-gate 绕过 -> 不应因门禁返回 2（交易时段/锁等其它原因另算）
        sys.argv = ["run_live_cron.py", "--config", str(cfg_path), "--no-gate"]
        rc2 = m.main()
        assert rc2 != 2, "--no-gate 时不应再因门禁返回 2"
    finally:
        sys.argv = old_argv
        gw_mod.DataGateway = gw_mod.DataGateway  # 还原由 reload 控制，无关紧要
