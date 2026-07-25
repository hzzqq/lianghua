"""数据网关缓存正确性回归。

核心修复：缓存仅在完整覆盖请求区间时命中，避免「先取窄区间、再取宽区间」
返回不完整的旧缓存数据；_cache_put 改为 upsert 刷新；新增 force_refresh。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from lianghua.data.gateway import DataGateway


def _gw():
    fd, db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return DataGateway(cache_db=db), db


def _clean(db):
    # Windows 下 sqlite 句柄释放与临时目录锁可能导致删除失败，属环境性，尽力清理即可
    try:
        os.remove(db)
    except OSError:
        pass


def test_partial_cache_refreshes_to_full():
    gw, db = _gw()
    narrow = gw.fetch("600519.SH", "2023-01-01", "2023-06-30")
    assert "source" in narrow.columns
    n_narrow = len(narrow)
    # 再取更宽区间：缓存只有 Jan-Jun，应重新拉取补齐到全年，而非返回子集
    wide = gw.fetch("600519.SH", "2023-01-01", "2023-12-31")
    assert len(wide) > n_narrow, "宽区间应补齐缺失数据，而非返回窄区间缓存子集"
    assert str(wide["date"].max()) >= "2023-12-29", "宽区间末端应覆盖到 12 月"
    _clean(db)


def test_full_coverage_cache_hit_stable():
    gw, db = _gw()
    a = gw.fetch("000300.SH", "2023-01-01", "2023-12-31")
    b = gw.fetch("000300.SH", "2023-01-01", "2023-12-31")
    assert len(a) == len(b), "完整覆盖应命中缓存且结果一致"
    _clean(db)


def test_force_refresh_no_error():
    gw, db = _gw()
    a = gw.fetch("510300.SH", "2023-01-01", "2023-06-30")
    b = gw.fetch("510300.SH", "2023-01-01", "2023-06-30", force_refresh=True)
    assert "source" in b.columns and len(b) == len(a)
    _clean(db)


def test_source_column_present():
    gw, db = _gw()
    df = gw.fetch("RB0.SHF", "2023-01-01", "2023-03-31", asset="future")
    assert "source" in df.columns
    assert df["source"].iloc[0] in ("cache", "csv", "akshare", "baostock", "demo")
    _clean(db)
