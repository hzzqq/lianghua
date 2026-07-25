"""optimize/param_search 与 data/sources 打磨测试：空采样/全NaN/数据质量门。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lianghua.optimize.param_search import (
    best_params,
    grid_search,
    random_search,
    summarize,
)
from lianghua.data.sources import (
    SyntheticSource,
    validate_ohlcv,
    fetch_from,
)


# ---------------- param_search ----------------
def _sig(df, n=10, **params):
    return pd.Series([0] * len(df))


def test_random_search_empty_list_raises():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        random_search(df, _sig, {"k": []}, n_iter=5)


def test_best_params_skips_nan():
    res = pd.DataFrame({
        "k": [1, 2, 3],
        "score": [float("nan"), 0.5, 0.9],
        "error": ["boom", "", ""],
    }).sort_values("score", ascending=False).reset_index(drop=True)
    bp = best_params(res)
    assert bp == {"k": 3}  # 跳过 NaN 行


def test_best_params_all_nan_returns_none():
    res = pd.DataFrame({"k": [1, 2], "score": [float("nan"), float("nan")], "error": ["x", "y"]})
    assert best_params(res) is None


def test_summarize_stats():
    res = pd.DataFrame({
        "k": [1, 2, 3],
        "score": [0.2, 0.9, float("nan")],
        "error": ["", "", "boom"],
    })
    s = summarize(res)
    assert s["n_total"] == 3
    assert s["n_valid"] == 2
    assert s["n_fail"] == 1
    assert s["best_score"] == pytest.approx(0.9)


def test_grid_search_runs():
    df = pd.DataFrame({"close": np.linspace(100, 110, 30)})
    res = grid_search(df, _sig, {"k": [1, 2, 3]})
    assert len(res) == 3
    assert "score" in res.columns


# ---------------- data/sources ----------------
def test_synth_start_after_end_no_crash():
    df = SyntheticSource().fetch("X", "2023-06-01", "2023-01-01")
    assert df is not None
    assert len(df) >= 2
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]


def test_validate_ohlcv_drops_nan_rows():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4),
        "open": [1, 2, np.nan, 4],
        "high": [1.1, 2.1, 3.1, 4.1],
        "low": [0.9, 1.9, 2.9, 3.9],
        "close": [1.0, 2.0, 3.0, 4.0],
        "volume": [10, 20, 30, 40],
    })
    out = validate_ohlcv(df)
    assert len(out) == 3  # 含 NaN 的一行被剔除


def test_validate_ohlcv_rejects_missing_columns():
    with pytest.raises(ValueError):
        validate_ohlcv(pd.DataFrame({"close": [1, 2]}))


def test_validate_ohlcv_clean_mixed_nonpositive_close():
    # 非正 close 行应被剔除（清洗），保留 1 行有效数据
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2),
        "open": [1, 1], "high": [1, 1], "low": [1, 1],
        "close": [0.0, 1.0], "volume": [1, 1],
    })
    out = validate_ohlcv(df)
    assert len(out) == 1


def test_validate_ohlcv_rejects_all_nonpositive_close():
    # 全部非正 close → 清洗后为空 → 抛 ValueError
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2),
        "open": [1, 1], "high": [1, 1], "low": [1, 1],
        "close": [0.0, 0.0], "volume": [1, 1],
    })
    with pytest.raises(ValueError):
        validate_ohlcv(df)


def test_akshare_source_does_not_crash():
    # 离线/无依赖时返回 None 或 DataFrame，均不应抛错
    out = fetch_from("akshare", "600000.SH", "2023-01-01", "2023-02-01")
    assert out is None or isinstance(out, pd.DataFrame)
