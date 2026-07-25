"""特征存储：新增 get_features/has/purge + 数据质量守卫回归测试。

覆盖：
- 非有限(NaN/inf)/非数值特征被拒绝写入（隐性数据污染修复）
- 缺 date/特征列、空 df/空 names 守卫
- get_features 面板对齐、has 可观测、purge 清理
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from lianghua.data.feature_store import FeatureStore


@pytest.fixture
def store():
    d = tempfile.mkdtemp()
    db = os.path.join(d, "f.db")
    yield FeatureStore(db)
    try:
        if os.path.exists(db):
            os.remove(db)
    except OSError:
        pass  # Windows 下 sqlite 文件句柄可能延迟释放


def _df(values=(55.0, 60.0), dates=("2023-01-02", "2023-01-03")):
    return pd.DataFrame({"date": list(dates), "rsi": list(values)})


def test_add_get_roundtrip(store):
    store.add("AAA", _df(), ["rsi"])
    s = store.get("AAA", "rsi")
    assert len(s) == 2
    assert s.iloc[0] == 55.0


def test_nonfinite_rejected(store):
    with pytest.raises(ValueError):
        store.add("AAA", _df(values=(55.0, np.nan)), ["rsi"])
    with pytest.raises(ValueError):
        store.add("AAA", _df(values=(55.0, np.inf)), ["rsi"])


def test_nonnumeric_rejected(store):
    with pytest.raises(ValueError):
        store.add("AAA", _df(values=("55.0", "abc")), ["rsi"])


def test_missing_columns_rejected(store):
    with pytest.raises(ValueError):
        store.add("AAA", pd.DataFrame({"rsi": [1.0]}), ["rsi"])  # 无 date
    with pytest.raises(ValueError):
        store.add("AAA", _df(), ["missing_col"])  # 缺特征列


def test_empty_inputs_rejected(store):
    with pytest.raises(ValueError):
        store.add("AAA", pd.DataFrame(), ["rsi"])
    with pytest.raises(ValueError):
        store.add("AAA", _df(), [])


def test_get_features_panel(store):
    df = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-03"],
        "rsi": [55.0, 60.0],
        "macd": [0.1, 0.2],
    })
    store.add("AAA", df, ["rsi", "macd"])
    panel = store.get_features("AAA")
    assert set(panel.columns) == {"rsi", "macd"}
    assert len(panel) == 2
    # 指定子集
    sub = store.get_features("AAA", ["rsi"])
    assert list(sub.columns) == ["rsi"]


def test_get_features_missing_symbol_empty(store):
    assert store.get_features("NOPE").empty
    assert store.get_features("NOPE", ["rsi"]).empty


def test_has(store):
    assert not store.has("AAA", "rsi")
    store.add("AAA", _df(), ["rsi"])
    assert store.has("AAA", "rsi")
    assert not store.has("AAA", "macd")


def test_purge(store):
    store.add("AAA", _df(), ["rsi"])
    store.add("BBB", _df(), ["rsi"])
    store.purge("AAA")
    assert not store.has("AAA", "rsi")
    assert store.has("BBB", "rsi")
    store.purge()
    assert not store.has("BBB", "rsi")
