"""DataGateway 健壮性 + 缓存内省能力测试。

用项目内临时缓存库（.pytest_tmp），避免污染根 data_cache.db，也规避 tmp_path fixture 的环境收尾问题。
"""
import os
import uuid

import pandas as pd
import pytest

from lianghua.data.gateway import DataGateway
from lianghua.core.assets import AssetType


@pytest.fixture
def gw():
    os.makedirs(".pytest_tmp", exist_ok=True)
    db = os.path.join(".pytest_tmp", f"gw_{uuid.uuid4().hex}.db")
    g = DataGateway(cache_db=db)
    yield g
    if os.path.exists(db):
        try:
            os.remove(db)
        except BaseException:
            # 沙箱 safe-delete 拦截可能抛 SystemExit/异常，清理失败不影响测试结论
            pass


def test_fetch_demo_stock_columns_and_source(gw):
    df = gw.fetch("600519.SH", "2024-01-01", "2024-01-10")
    assert isinstance(df, pd.DataFrame)
    for c in ["date", "open", "high", "low", "close", "volume", "source"]:
        assert c in df.columns
    assert not df.empty
    # 离线（无 akshare）应降级到 demo，但仍标注来源可追溯
    assert df["source"].iloc[0] in {"demo", "csv", "akshare", "baostock"}


def test_fetch_invalid_date_range_raises(gw):
    with pytest.raises(ValueError):
        gw.fetch("600519.SH", "2024-01-10", "2024-01-01")


def test_fetch_bad_date_format_raises(gw):
    with pytest.raises(ValueError):
        gw.fetch("600519.SH", "not-a-date", "2024-01-10")


def test_fetch_minute_bars_guard(gw):
    with pytest.raises(ValueError):
        gw.fetch_minute("600519.SH", "2024-01-02", bars=0)
    with pytest.raises(ValueError):
        gw.fetch_minute("600519.SH", "2024-01-02", bars=-5)


def test_fetch_minute_returns_expected_shape(gw):
    df = gw.fetch_minute("600519.SH", "2024-01-02", bars=10)
    assert len(df) == 10
    assert "datetime" in df.columns
    assert list(df.columns).count("close") == 1


def test_list_cached_after_fetch(gw):
    # 用真实源（mock）取数，验证 list_cached 能反映已落库的行情
    dates = pd.bdate_range("2024-01-01", "2024-01-10")
    real = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0, "volume": 1e6,
    })
    gw._from_akshare = lambda symbol, start, end, asset: real
    gw.fetch("600519.SH", "2024-01-01", "2024-01-10")
    cached = gw.list_cached()
    assert "600519.SH" in cached
    info = cached["600519.SH"]
    assert info["rows"] > 0
    assert info["source"] == "akshare"
    assert info["min"] <= info["max"]


def test_with_timeout_records_error_and_returns_none(gw):
    def boom():
        raise RuntimeError("network down")

    res = gw._with_timeout(boom, timeout=2)
    assert res is None
    assert gw._last_errors and "network down" in gw._last_errors[-1]


def test_list_cached_filter_by_asset(gw):
    # 用真实源（mock）取数，演示数据不落缓存，故须真实源才能让 list_cached 有内容
    dates = pd.bdate_range("2024-01-01", "2024-01-10")
    real = pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0, "volume": 1e6,
    })
    gw._from_akshare = lambda symbol, start, end, asset: real
    gw.fetch("600519.SH", "2024-01-01", "2024-01-10")
    # 未拉取期权时，按 OPTION 过滤应为空
    opts = gw.list_cached(asset=AssetType.OPTION)
    assert "600519.SH" not in opts
    # 按非期权资产过滤应返回 bars 中的标的
    stocks = gw.list_cached(asset=AssetType.STOCK)
    assert "600519.SH" in stocks
