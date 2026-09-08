# -*- coding: utf-8 -*-
"""腾讯行情真实源接入测试（第十七轮：真实数据通路修复）。

锁死两类不变量：
1. 腾讯公开行情接口在受限网络（沙箱代理白名单外 web.ifzq 被 501 时）仍能经
   ``proxy.finance.qq.com`` 拿到真实日 K / 实时快照——这是「框架实际能用真实数据」
   的根基，回归即意味着「真实行情又取不到了 → 静默降级演示数据 → 回测结论无意义」。
2. 网关在 AKShare / BaoStock 不可用时，应兜底到腾讯真实源（``last_was_demo=False``），
   而不是降级演示（假）数据。
"""
from __future__ import annotations

import pandas as pd
import pytest

from lianghua.data import tencent as tx
from lianghua.data.gateway import DataGateway


def test_code_map():
    assert tx._to_tencent_code("600519.SH") == "sh600519"
    assert tx._to_tencent_code("000001.SZ") == "sz000001"
    assert tx._to_tencent_code("510300.SH") == "sh510300"
    # 场外开放式基金腾讯无公开行情，必须返回 None（由上层降级其它源）
    assert tx._to_tencent_code("110011.OF") is None


def test_fetch_daily_real_and_shape():
    df = tx.fetch_daily("600519.SH", "2024-01-01", "2024-03-01", timeout=12)
    assert isinstance(df, pd.DataFrame) and not df.empty
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    # 真实价格应合理（茅台不可能 < 1 元）
    assert df["close"].iloc[0] > 1.0
    assert (df["high"] >= df["low"]).all()


def test_fetch_daily_full_range_no_duplicate():
    """长区间翻页：腾讯 start 参数对长区间被忽略，必须靠 end 前移翻页；
    且翻页/源返回不能产生重复日期（否则写入端撞主键）。"""
    df = tx.fetch_daily("600519.SH", "2023-01-01", "2025-06-01", timeout=15)
    assert isinstance(df, pd.DataFrame) and not df.empty
    assert not df["date"].duplicated().any(), "翻页产生重复日期会撞主键"
    assert len(df) > 300, "长区间未完整翻页"
    assert df["date"].min() <= "2023-01-31"
    assert df["date"].max() >= "2025-05-01"


def test_fetch_quote_real():
    px = tx.fetch_quote("600519.SH")
    assert px is not None and px > 1.0


def test_gateway_falls_back_to_tencent_not_demo():
    """核心不变量：真实源全挂时，腾讯兜底必须拿到真实数据，绝不静默降级演示。"""
    g = DataGateway()
    df = g.fetch("600519.SH", "2024-01-01", "2024-06-01", timeout=12)
    assert df is not None and not df.empty
    assert g.last_was_demo is False, "腾讯兜底应拿到真实数据，不应降级演示(假)数据"
    assert g.last_source in ("tencent", "akshare", "baostock", "cache")


def test_live_quote_real_source_preferred():
    g = DataGateway()
    q = g.live_quote("600519.SH")
    assert q["price"] == q["price"] and q["price"] is not None  # 非 NaN
    assert q["source"] != "demo", "实时报价应优先真实源（akshare/tencent）"
