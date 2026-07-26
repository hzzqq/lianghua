"""R43: zscore 鲁棒化与能力验收。"""
import numpy as np
import pandas as pd

from lianghua.factor.zscore import (
    winsorize, zscore, robust_zscore, winsorize_group, neutralize,
)


def test_winsorize_basic():
    s = pd.Series(list(range(100)) + [100000])
    w = winsorize(s)
    assert w.max() < 100000            # 极端高位被截断
    assert w.min() >= s.quantile(0.01) - 1e-9  # 低位亦被缩尾
    assert np.isfinite(w).all()


def test_winsorize_empty_returns_input():
    s = pd.Series([], dtype=float)
    assert winsorize(s).empty


def test_winsorize_all_nan_no_crash():
    s = pd.Series([np.nan] * 5)
    r = winsorize(s)
    assert len(r) == 5


def test_winsorize_quantiles_autoswap():
    # lower>upper 时自动交换，不抛错且产出有限结果
    w = winsorize(pd.Series([1, 2, 3, 4, 5, 1000]), lower=0.9, upper=0.1)
    assert np.isfinite(w).all()
    assert w.max() < 1000


def test_zscore_constant_series_zero():
    # 常数列原本会被 1e-12 除出巨大值，现应返回全 0
    s = pd.Series([5.0] * 10)
    z = zscore(s)
    assert np.allclose(z, 0.0)


def test_zscore_nonfinite_safe():
    s = pd.Series([1.0, np.nan, 3.0])
    z = zscore(s)
    assert np.isfinite(z.iloc[0]) and np.isnan(z.iloc[1])


def test_robust_zscore_outlier_robust():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])  # 单离群点
    rz = robust_zscore(s)
    # 中位数附近点应接近 0，离群点被压缩
    assert np.isfinite(rz).all()
    assert abs(rz.iloc[4]) < 1.0  # 中值点


def test_robust_zscore_constant_zero():
    s = pd.Series([2.0] * 8)
    assert np.allclose(robust_zscore(s), 0.0)


def test_winsorize_group_per_group():
    s = pd.Series([1, 2, 3, 1, 2, 100])  # 第二组末尾含组内极端值
    g = pd.Series(["a", "a", "a", "b", "b", "b"])
    w = winsorize_group(s, g)
    # 组 b 的 100 应被组内上分位截断，而非被组 a 的分布影响
    assert w.max() < 100
    assert len(w) == 6


def test_neutralize_basic():
    f = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    ind = pd.Series(["x", "x", "x", "y", "y", "y"])
    n = neutralize(f, ind)
    assert np.isfinite(n).all()
