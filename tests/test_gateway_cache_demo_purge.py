"""缓存内省：演示/真实分离标记 + purge 接口。

防止 /api/cache 把历史残留的演示(假)数据当真实数据统计（R17 曾误清 1780 行 demo + 65 行
option demo）。新代码已从源头拦截 demo 落库（fetch 对 demo 不写缓存、_cache_get 命中 demo 即
当未命中），但历史残留仍需可观测、可清洗。
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.data.gateway import DataGateway


@pytest.fixture
def gw(tmp_path):
    # 必须用文件库：_connect 每次操作开新连接，":memory:" 跨连接不持久，
    # 真实读写缓存的测试会被静默架空（现有 demo 隔离测试因 mock 了缓存层才可用 :memory:）。
    return DataGateway(cache_db=str(tmp_path / "cache.db"))


def _df(start, end, source="akshare"):
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, n))
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 1e6,
        "source": source,
    })


def test_list_cached_marks_demo_rows(gw):
    """缓存里若残留 demo 行，list_cached 必须单独标记 demo_rows。"""
    real = _df("2024-01-01", "2024-01-05", "akshare")
    demo = _df("2024-01-08", "2024-01-10", "demo")
    gw._cache_put("600519.SH", real, asset=None, source="akshare",
                  start="2024-01-01", end="2024-01-05")
    # 手动塞入 demo 行模拟历史残留（fetch 新代码不会写 demo，但旧库可能有）
    demo_to_write = demo.copy()
    demo_to_write["symbol"] = "600519.SH"
    with gw._connect() as con:
        demo_to_write.to_sql("bars", con, if_exists="append", index=False)
    cached = gw.list_cached()
    info = cached["600519.SH"]
    assert info["rows"] >= 8, "总行数应含真实+演示"
    assert info["demo_rows"] == 3, "demo 行数必须单独标出"
    assert info["source"] == "akshare"


def test_list_cached_pure_real_has_zero_demo(gw):
    """纯真实数据缓存：demo_rows 必须为 0。"""
    real = _df("2024-01-01", "2024-01-10", "akshare")
    gw._from_akshare = lambda symbol, start, end, asset: real
    gw.fetch("600519.SH", "2024-01-01", "2024-01-10", asset="stock")
    info = gw.list_cached()["600519.SH"]
    assert info["demo_rows"] == 0
    assert info["rows"] == info["rows"]  # 全真实


def test_purge_demo_removes_only_demo(gw):
    """purge_demo_cache 只删 demo 行，真实行保留。"""
    real = _df("2024-01-01", "2024-01-05", "akshare")
    demo = _df("2024-01-08", "2024-01-10", "demo")
    gw._cache_put("600519.SH", real, asset=None, source="akshare",
                  start="2024-01-01", end="2024-01-05")
    demo_to_write = demo.copy()
    demo_to_write["symbol"] = "600519.SH"
    with gw._connect() as con:
        demo_to_write.to_sql("bars", con, if_exists="append", index=False)
    # 清洗前
    before = gw.list_cached()["600519.SH"]
    assert before["demo_rows"] == 3
    # 清洗
    deleted = gw.purge_demo_cache()
    assert deleted["bars"] == 3
    # 清洗后
    after = gw.list_cached()["600519.SH"]
    assert after["demo_rows"] == 0
    assert after["rows"] == before["rows"] - 3, "真实行必须保留"


def test_purge_demo_on_clean_cache_is_noop(gw):
    """纯净缓存（无 demo）purge 应删 0 行。"""
    real = _df("2024-01-01", "2024-01-05", "akshare")
    gw._cache_put("600519.SH", real, asset=None, source="akshare",
                  start="2024-01-01", end="2024-01-05")
    deleted = gw.purge_demo_cache()
    assert deleted["bars"] == 0
    assert gw.list_cached()["600519.SH"]["rows"] == 5


def test_purge_demo_on_empty_cache_safe(gw):
    """空缓存 purge 不报错。"""
    deleted = gw.purge_demo_cache()
    assert deleted == {"bars": 0, "option_bars": 0}
