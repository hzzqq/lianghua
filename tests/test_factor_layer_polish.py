"""Cycle 52：factor.layer 打磨——ic_series 性能/兼容性 + qcut 崩溃 + 多空 NaN 守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.factor.layer import (
    ic_series, quantile_returns, long_short_by_factor, long_short_detail,
)


def _factor_returns(n=300, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.Series(rng.normal(0, 1, n))
    r = pd.Series(f * 0.1 + rng.normal(0, 1, n) * 0.5)
    return f, r


def test_ic_series_efficient_and_finite():
    f, r = _factor_returns()
    s = ic_series(f, r, window=20)
    assert len(s) > 0
    assert s.notna().all()
    # 兼容旧调用 lags= 别名
    s2 = ic_series(f, r, lags=20)
    assert np.allclose(s.values, s2.values)


def test_ic_series_validates_window():
    f, r = _factor_returns(n=50)
    with pytest.raises(ValueError):
        ic_series(f, r, window=1)


def test_quantile_returns_constant_factor_no_crash():
    # 隐性修复：常数因子不应让 qcut 抛 ValueError
    n = 200
    f = pd.Series([0.5] * n)
    r = pd.Series(np.random.default_rng(1).normal(0, 1, n))
    q = quantile_returns(f, r, n=5)
    assert not q.empty
    assert np.isfinite(q.iloc[0])


def test_quantile_returns_low_cardinality_handled():
    # 低基数因子（仅 3 个离散值）请求 5 组不应崩溃
    f = pd.Series(np.tile([0.0, 1.0, 2.0], 60))
    r = pd.Series(np.random.default_rng(2).normal(0, 0.1, 180))
    q = quantile_returns(f, r, n=5)
    assert q.notna().all()


def test_long_short_nan_when_insufficient_groups():
    # 隐性修复：不足两组时不应静默给出错误价差
    f = pd.Series([0.5] * 100)  # 常数 -> 单组
    r = pd.Series(np.random.default_rng(3).normal(0, 1, 100))
    ls = long_short_by_factor(f, r, n=5)
    assert np.isnan(ls)
    det = long_short_detail(f, r, n=5)
    assert np.isnan(det["spread"])
    assert det["n_groups"] < 2


def test_long_short_detail_structure():
    f, r = _factor_returns(seed=5)
    det = long_short_detail(f, r, n=5)
    assert set(det) == {"top", "bottom", "spread", "n_groups"}
    assert det["n_groups"] == 5
    assert np.isclose(det["spread"], det["top"] - det["bottom"])
    assert np.isfinite(det["spread"])


def test_inf_filtered_in_quantile():
    f, r = _factor_returns()
    f = f.copy()
    f.iloc[0] = np.inf
    q = quantile_returns(f, r, n=5)
    assert q.notna().all()
