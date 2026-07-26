"""数据层（data）：resample 成交量别名修复 + 特征存储对齐/内省 + universe 成员判定。

覆盖（续做轮 55，data 集群）：
- 隐性 bug：resample_ohlcv 的 _VOL_ALIASES 含 ' turnover'(前导空格) 导致 'turnover' 列
  无法识别为成交量；修复后 'turnover' 应被识别并规范为 'volume'。
- 隐性 bug：FeatureStore.add 对 Timestamp / 带时分秒的 date 未规范为 %Y-%m-%d，
  会导致与行情表 join 不上；修复后统一规范。
- 新需求：FeatureStore.align 把缓存特征按 date 合并到价格序列。
- 新需求：FeatureStore.overview 特征覆盖概况（可观测性）。
- 新需求：universe.in_universe / union 成分判定与去重并集。
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from lianghua.data.feature_store import FeatureStore
from lianghua.data import resample as rs
from lianghua.data import universe


# ---------------- resample 成交量别名 ----------------
def test_resample_recognizes_turnover_alias():
    dates = pd.date_range("2023-01-02", periods=6, freq="B")
    df = pd.DataFrame({
        "open": np.linspace(10, 11, 6),
        "high": np.linspace(10.5, 11.6, 6),
        "low": np.linspace(9.5, 10.4, 6),
        "close": np.linspace(10.1, 11.1, 6),
        "turnover": np.linspace(1e6, 2e6, 6),
    }, index=dates)
    out = rs.resample_ohlcv(df, rule="W")
    assert not out.empty
    assert "volume" in out.columns  # turnover 被识别并规范为 volume
    assert out["volume"].iloc[0] > 0


def test_resample_volume_typo_rejected():
    # 带前导空格的非法列名不应被当作成交量别名
    dates = pd.date_range("2023-01-02", periods=4, freq="B")
    df = pd.DataFrame({
        "open": [1, 2, 3, 4], "high": [1, 2, 3, 4],
        "low": [1, 2, 3, 4], "close": [1, 2, 3, 4],
        " turnover": [1, 2, 3, 4],
    }, index=dates)
    out = rs.resample_ohlcv(df, rule="W")
    assert "volume" not in out.columns  # 前导空格别名不匹配


# ---------------- FeatureStore 规范化 / 对齐 / 内省 ----------------
@pytest.fixture
def store():
    d = tempfile.mkdtemp()
    db = os.path.join(d, "f.db")
    yield FeatureStore(db)
    try:
        if os.path.exists(db):
            os.remove(db)
    except OSError:
        pass


def _df_with_timestamp_dates():
    dates = pd.to_datetime(["2023-01-02", "2023-01-03"])
    return pd.DataFrame({
        "date": dates,  # Timestamp 类型，触发器：未规范会存成带时分秒字符串
        "rsi": [55.0, 60.0],
    })


def test_add_normalizes_timestamp_date(store):
    store.add("AAA", _df_with_timestamp_dates(), ["rsi"])
    s = store.get("AAA", "rsi")
    assert list(s.index) == ["2023-01-02", "2023-01-03"]  # 规范为 %Y-%m-%d


def test_align_merges_features_onto_price(store):
    store.add("AAA", _df_with_timestamp_dates(), ["rsi"])
    price = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-03", "2023-01-04"],
        "close": [10.0, 10.5, 11.0],
    })
    merged = store.align(price, "AAA", ["rsi"])
    assert "rsi" in merged.columns
    # 前两行有特征值，末行（无特征）应为 NaN（左连接）
    assert merged["rsi"].iloc[0] == 55.0
    assert pd.isna(merged["rsi"].iloc[2])


def test_align_requires_date_column(store):
    with pytest.raises(ValueError):
        store.align(pd.DataFrame({"close": [1.0]}), "AAA", ["rsi"])


def test_overview_insight(store):
    store.add("AAA", _df_with_timestamp_dates(), ["rsi"])
    store.add("AAA", pd.DataFrame({
        "date": ["2023-01-02", "2023-01-03"], "macd": [0.1, 0.2]}), ["macd"])
    ov = store.overview("AAA")
    assert ov == {"rsi": 2, "macd": 2}
    full = store.overview()
    assert "AAA" in full and full["AAA"]["features"] == 2 and full["AAA"]["dates"] == 2


# ---------------- universe 成员判定 ----------------
def test_in_universe_and_union():
    assert universe.in_universe("白酒", "600519.SH")
    assert not universe.in_universe("白酒", "000001.SZ")
    with pytest.raises(ValueError):
        universe.in_universe("未知", "600519.SH")
    u = universe.union(["白酒", "银行"])
    assert "600519.SH" in u and "601398.SH" in u
    assert len(u) == len(set(u))  # 去重
    allu = universe.union()
    assert len(allu) == len(set(allu))
