"""网关演示(假)数据缓存隔离测试。

修复的隐性正确性 bug：
- 首次 fetch 真实源失败降级到演示(假)数据后，该假数据会被写入本地缓存；
- 后续 fetch 直接命中缓存、永不重试真实源，用户从此永久看到假数据而毫无察觉。
本测试验证：演示数据既不会被写入缓存，也不会从缓存中被取出（永远重新尝试真实源）。
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.data.gateway import DataGateway


@pytest.fixture
def gw():
    return DataGateway(cache_db=":memory:")


def _real_df(start="2024-01-01", end="2024-01-10"):
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, n))
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 1e6,
    })


def test_demo_cache_hit_is_ignored(gw):
    """缓存里若是演示数据，fetch 必须忽略它并重新走真实源。"""
    demo = _real_df().copy()
    demo["source"] = "demo"

    # 模拟“历史上曾缓存过 demo”的场景
    gw._cache_get = lambda *a, **k: demo
    # 真实源这次可用
    gw._from_akshare = lambda symbol, start, end, asset: _real_df()

    df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock")
    assert df["source"].iloc[0] == "akshare", "缓存中的 demo 不应被取出，应改取真实源"
    assert not gw.last_was_demo


def test_demo_is_not_persisted(gw):
    """真实源失败时拿到的演示数据不应被写入缓存（_cache_put 不应以 demo 落库）。"""
    put_calls = []

    def fake_put(symbol, df, asset, source="unknown"):
        put_calls.append(source)

    gw._cache_put = fake_put
    gw._from_akshare = lambda symbol, start, end, asset: (_ for _ in ()).throw(
        ConnectionError("down"))
    gw._from_baostock = lambda symbol, start, end: (_ for _ in ()).throw(
        ConnectionError("down"))

    df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock")
    assert gw.last_was_demo
    assert "demo" not in put_calls, "演示数据不应被写入缓存"


def test_real_data_still_cached(gw):
    """真实数据仍应正常缓存（行为不退化）。"""
    put_calls = []

    def fake_put(symbol, df, asset, source="unknown"):
        put_calls.append(source)

    gw._cache_put = fake_put
    gw._from_akshare = lambda symbol, start, end, asset: _real_df()

    gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock")
    assert "akshare" in put_calls, "真实数据应照常写入缓存"
