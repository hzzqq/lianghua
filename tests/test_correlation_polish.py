"""Cycle 50：risk.correlation 打磨——inf 泄漏/NaN 传播/窗口校验 + 最近正定相关矩阵。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.risk.correlation import (
    corr_matrix, average_correlation, rolling_corr, is_pd, nearest_pd_correlation,
)


def _rets(n=120, cols=("A", "B", "C")):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 0.01, (n, len(cols))), columns=cols)


def test_corr_matrix_filters_inf_and_has_unit_diag():
    # 隐性修复：inf 不应污染相关性矩阵
    df = _rets()
    df.loc[df.index[0], "A"] = np.inf
    df.loc[df.index[1], "B"] = -np.inf
    c = corr_matrix(df)
    assert np.all(np.isfinite(c.values))
    assert np.allclose(np.diag(c.values), 1.0)


def test_corr_matrix_rejects_too_few_assets_or_rows():
    with pytest.raises(ValueError):
        corr_matrix(pd.DataFrame({"A": [1, 2, 3]}))
    with pytest.raises(ValueError):
        corr_matrix(pd.DataFrame({"A": [1.0], "B": [2.0]}))


def test_average_correlation_ignores_nan_pairs():
    # 隐性修复：含常数列（相关性 NaN）时不应传播 NaN
    df = _rets()
    df["CONST"] = 0.5  # 常数资产
    ac = average_correlation(df)
    assert np.isfinite(ac)
    ac_no_const = average_correlation(df[["A", "B", "C"]])
    assert np.isfinite(ac_no_const)


def test_rolling_corr_validates_window_and_short_input():
    a = pd.Series(np.random.default_rng(1).normal(0, 1, 50))
    b = pd.Series(np.random.default_rng(2).normal(0, 1, 50))
    with pytest.raises(ValueError):
        rolling_corr(a, b, window=1)
    # 长度不足窗口 -> 空序列而非全 NaN 崩溃
    rc = rolling_corr(a, b, window=200)
    assert isinstance(rc, pd.Series) and rc.empty
    rc2 = rolling_corr(a, b, window=20)
    assert rc2.notna().any()


def test_is_pd_detects_indefinite():
    indef = pd.DataFrame([[1.0, 1.2], [1.2, 1.0]])  # 特征值含负
    assert is_pd(indef) is False
    assert is_pd(pd.DataFrame(np.eye(3))) is True


def test_nearest_pd_correlation_repairs():
    # 非半正定相关矩阵 -> 修复后正定且单位对角
    bad = pd.DataFrame([[1.0, 1.2, 0.3], [1.2, 1.0, 0.4], [0.3, 0.4, 1.0]],
                       index=["A", "B", "C"], columns=["A", "B", "C"])
    assert is_pd(bad) is False
    fixed = nearest_pd_correlation(bad)
    assert is_pd(fixed) is True
    assert np.allclose(np.diag(fixed.values), 1.0, atol=1e-9)
    assert fixed.shape == (3, 3)


def test_nearest_pd_correlation_on_real_correlation():
    # 真实（近似）相关矩阵修复后仍保持结构
    c = corr_matrix(_rets())
    fixed = nearest_pd_correlation(c)
    assert is_pd(fixed) is True
    assert np.allclose(np.diag(fixed.values), 1.0, atol=1e-9)
