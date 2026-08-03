"""缓存覆盖判定与 sqlite 连接生命周期。

这两条都是看代码看不出来、只有真跑起来才暴露的坑：

1. 覆盖判定原本靠"落库数据的首尾日期"反推。A股一周只有 5 个交易日，
   用户请求 2024-01-01~2024-03-31 这种整季度区间时，数据实际只有 01-02~03-29，
   反推法永远判定"未覆盖"→ 每次 fetch 都重新打真实源。轻则慢，重则在断网/限流时
   把一份完好的真实缓存白白丢掉、降级成演示（假）数据。
   改为单独记录"已向真实源问过的窗口"（cache_range 表）。
2. ``with sqlite3.connect(...) as con`` 只提交事务、**不关闭连接**。每次读写泄漏一个
   句柄，长跑的实盘进程 fd 只增不减，Windows 上缓存库文件还会被永久占用。
"""
from __future__ import annotations

import gc
import os
import sqlite3
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lianghua.core.assets import AssetType  # noqa: E402
from lianghua.data.gateway import DataGateway  # noqa: E402


def _bars(start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0,
    })


@pytest.fixture()
def cache_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    gc.collect()
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _gw_with_source(cache_db, df: pd.DataFrame, calls: list) -> DataGateway:
    gw = DataGateway(cache_db=cache_db)

    def _src(symbol, start, end, asset=None):
        calls.append((start, end))
        return df.copy()

    gw._from_akshare = _src
    return gw


def test_cache_hits_even_when_range_edges_are_non_trading_days(cache_db):
    """请求区间两端是元旦/周末时，二次 fetch 也必须命中缓存，而不是重新打真实源。"""
    calls: list = []
    gw = _gw_with_source(cache_db, _bars("2024-01-02", "2024-03-29"), calls)

    first = gw.fetch("600000.SH", "2024-01-01", "2024-03-31", asset=AssetType.STOCK)
    assert len(first) > 0 and gw.last_source == "akshare"

    second = gw.fetch("600000.SH", "2024-01-01", "2024-03-31", asset=AssetType.STOCK)
    assert gw.last_source == "cache", "整季度区间的第二次 fetch 应命中缓存"
    assert len(calls) == 1, f"真实源被重复调用: {calls}"
    pd.testing.assert_frame_equal(
        first.drop(columns=["source"]), second.drop(columns=["source"]))


def test_cache_miss_when_asking_beyond_known_range(cache_db):
    """只问过 Q1 就只敢声称覆盖 Q1；请求扩到 Q2 必须重新拉取，不能拿半截数据顶包。"""
    calls: list = []
    gw = _gw_with_source(cache_db, _bars("2024-01-02", "2024-03-29"), calls)

    gw.fetch("600000.SH", "2024-01-01", "2024-03-31", asset=AssetType.STOCK)
    gw._from_akshare = lambda symbol, start, end, asset=None: _bars("2024-01-02", "2024-06-28")

    out = gw.fetch("600000.SH", "2024-01-01", "2024-06-30", asset=AssetType.STOCK)
    assert gw.last_source == "akshare", "缓存未覆盖的区间不应命中缓存"
    assert str(out["date"].max()) >= "2024-06-01"


def test_force_refresh_still_bypasses_cache(cache_db):
    calls: list = []
    gw = _gw_with_source(cache_db, _bars("2024-01-02", "2024-03-29"), calls)

    gw.fetch("600000.SH", "2024-01-01", "2024-03-31", asset=AssetType.STOCK)
    gw.fetch("600000.SH", "2024-01-01", "2024-03-31", asset=AssetType.STOCK,
             force_refresh=True)
    assert len(calls) == 2 and gw.last_source == "akshare"


def test_repeated_fetch_does_not_leak_sqlite_handles(cache_db):
    """反复读写缓存后缓存库必须能被删除；删不掉即等于连接句柄泄漏。"""
    calls: list = []
    gw = _gw_with_source(cache_db, _bars("2024-01-02", "2024-03-29"), calls)
    for _ in range(5):
        gw.fetch("600000.SH", "2024-01-01", "2024-03-31", asset=AssetType.STOCK)
    gw.list_cached()

    del gw
    gc.collect()
    os.remove(cache_db)  # 泄漏时这里抛 PermissionError(WinError 32)
    assert not os.path.exists(cache_db)
    open(cache_db, "wb").close()  # 交还给 fixture 清理


def test_adjacent_ranges_are_merged_instead_of_wiping_each_other(cache_db):
    """在 UI 上来回切时间区间不该把上一段缓存整块删掉。

    取 Q1 → 取 Q2 → 回看 Q1，第三次必须命中缓存；
    并且合并后的 Q1+Q2 应能直接满足"跨两个季度"的请求。
    """
    calls: list = []
    gw = DataGateway(cache_db=cache_db)

    def _src(symbol, start, end, asset=None):
        calls.append((start, end))
        return _bars(start, end)

    gw._from_akshare = _src

    gw.fetch("600000.SH", "2024-01-01", "2024-03-31", asset=AssetType.STOCK)
    gw.fetch("600000.SH", "2024-04-01", "2024-06-30", asset=AssetType.STOCK)

    gw.fetch("600000.SH", "2024-01-01", "2024-03-31", asset=AssetType.STOCK)
    assert gw.last_source == "cache", f"回看 Q1 应命中缓存，实际重新拉取: {calls}"

    both = gw.fetch("600000.SH", "2024-01-01", "2024-06-30", asset=AssetType.STOCK)
    assert gw.last_source == "cache", "相邻区间合并后应能覆盖跨季度请求"
    assert len(both) == len(_bars("2024-01-01", "2024-06-30")), "合并后数据不应有缺口"
    assert len(calls) == 2, f"真实源只应被调用两次，实际: {calls}"


def test_disjoint_ranges_do_not_claim_to_cover_the_gap(cache_db):
    """两段区间之间有缺口时，绝不能谎称已覆盖整段——那会把缺口数据静默吞掉。"""
    gw = DataGateway(cache_db=cache_db)
    gw._from_akshare = lambda symbol, start, end, asset=None: _bars(start, end)

    gw.fetch("600000.SH", "2024-01-01", "2024-01-31", asset=AssetType.STOCK)
    gw.fetch("600000.SH", "2024-06-01", "2024-06-30", asset=AssetType.STOCK)

    pulled: list = []

    def _src(symbol, start, end, asset=None):
        pulled.append((start, end))
        return _bars(start, end)

    gw._from_akshare = _src
    out = gw.fetch("600000.SH", "2024-01-01", "2024-06-30", asset=AssetType.STOCK)
    assert pulled, "中间 2~5 月从未拉取过，不能直接命中缓存"
    assert len(out) == len(_bars("2024-01-01", "2024-06-30"))


def test_legacy_cache_without_range_table_still_usable(cache_db):
    """老缓存库没有 cache_range 表：退回按数据首尾判断，仍能命中，不能直接报错。"""
    con = sqlite3.connect(cache_db)
    con.execute("""CREATE TABLE bars (symbol TEXT, date TEXT, open REAL, high REAL,
                   low REAL, close REAL, volume REAL, source TEXT,
                   PRIMARY KEY (symbol, date))""")
    legacy = _bars("2024-01-01", "2024-03-31")
    legacy["symbol"] = "600000.SH"
    legacy["source"] = "akshare"
    legacy.to_sql("bars", con, if_exists="append", index=False)
    con.commit()
    con.close()

    gw = DataGateway(cache_db=cache_db)
    gw._from_akshare = lambda *a, **k: pytest.fail("老缓存已覆盖该区间，不应打真实源")
    out = gw.fetch("600000.SH", "2024-01-02", "2024-03-29", asset=AssetType.STOCK)
    assert gw.last_source == "cache" and len(out) > 0
