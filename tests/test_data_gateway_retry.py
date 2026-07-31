"""数据网关可靠性测试：真实源重试/退避 + 演示(假)数据降级可观测。

覆盖的隐性问题：
- 真实源因瞬时网络抖动偶发失败时，应自动重试而非立即降级到假数据；
- 重试耗尽最终落到演示数据时必须留下可观测痕迹（日志 warning + last_warnings()），
  避免用户在不经意间拿假数据跑回测/分析。

注：使用 :memory: 缓存，避免落盘与删除（沙箱禁止删除项目文件）。
"""
import logging

import numpy as np
import pandas as pd
import pytest

from lianghua.data.gateway import DataGateway


@pytest.fixture
def gw():
    # 内存缓存：不落盘、不触发删除拦截；缓存未命中由 mock 真实源接管
    return DataGateway(cache_db=":memory:")


def _fake_ohlc(start="2024-01-01", end="2024-01-10"):
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rng = np.random.default_rng(1)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, n))
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 1e6,
    })


def test_retry_succeeds_after_transient_failure(gw):
    """真实源前 2 次抛网络异常、第 3 次成功 → 重试后应拿到真实数据。"""
    calls = {"n": 0}

    def flaky(symbol, start, end, asset):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("simulated transient network error")
        return _fake_ohlc()

    gw._from_akshare = flaky
    df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10",
                  asset="stock", retries=2, backoff=0)
    assert calls["n"] == 3, "应重试到第三次才成功"
    assert not gw.last_was_demo, "重试成功不应落到演示数据"
    assert gw.last_source == "akshare"
    assert df["source"].iloc[0] == "akshare"
    assert len(df) > 0


def test_no_retry_falls_to_demo_with_observability(gw):
    """真实源始终失败且无重试 → 必须落到演示数据并在 last_warnings 留下原因。"""

    def always_fail(symbol, start, end, asset=None):
        raise ConnectionError("persistent failure")

    gw._from_akshare = always_fail
    gw._from_baostock = always_fail
    df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10",
                  asset="stock", retries=0, backoff=0)
    assert gw.last_was_demo, "真实源失败且无重试应降级到演示数据"
    assert df["source"].iloc[0] == "demo"
    warns = gw.last_warnings()
    assert warns, "降级到假数据必须在 last_warnings 留下原因"
    assert any("persistent failure" in w for w in warns), \
        "降级原因应包含真实源失败详情"


def test_demo_fallback_emits_warning_log(gw, caplog):
    """降级到演示数据时应打印 warning 日志，便于运维排障。"""

    def always_fail(symbol, start, end, asset):
        raise ConnectionError("persistent failure")

    gw._from_akshare = always_fail
    gw._from_baostock = always_fail
    with caplog.at_level(logging.WARNING):
        gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock")
    assert any("演示（假）数据" in r.message for r in caplog.records), \
        "降级演示数据应打 warning 日志"


def test_retry_clears_transient_errors_on_success(gw):
    """瞬时失败后成功，应清掉该源之前的失败记录，避免陈旧原因误导。"""
    calls = {"n": 0}

    def flaky(symbol, start, end, asset):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        return _fake_ohlc()

    gw._from_akshare = flaky
    gw.fetch("600000.SH", "2024-01-01", "2024-01-10",
             asset="stock", retries=1, backoff=0)
    assert not gw.last_was_demo
    # 成功路径下不应残留该源的 transient 失败记录
    assert not any("transient" in w for w in gw.last_warnings())
