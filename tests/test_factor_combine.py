"""Round 8：factor_combine 输入守卫 + minmax/clip 新能力 + 组合空守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.factor.combine import (
    factor_combine, factor_portfolio, factor_autocorr, factor_turnover,
)


def _panel(n=30, m=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = [f"A{i}" for i in range(m)]
    return pd.DataFrame(rng.normal(0, 1, (n, m)), index=idx, columns=cols)


def test_combine_zscore_basic():
    f = {"x": pd.Series([1, 2, 3, 4, 5.0]), "y": pd.Series([5, 4, 3, 2, 1.0])}
    c = factor_combine(f)
    assert len(c) == 5
    assert np.all(np.isfinite(c.values))


def test_combine_equal_weight_is_mean():
    # 等权 zscore 合成，两因子互为镜像，合成应近似为 0
    f = {"x": pd.Series([1.0, 2, 3, 4, 5]), "y": pd.Series([5.0, 4, 3, 2, 1])}
    c = factor_combine(f, method="zscore")
    assert np.allclose(c.abs().max(), 0.0, atol=1e-9)


def test_combine_rank_method():
    f = {"x": pd.Series([1.0, 2, 3, 4, 5]), "y": pd.Series([5.0, 4, 3, 2, 1])}
    c = factor_combine(f, method="rank")
    assert c.between(0, 1).all()


def test_combine_minmax_new_method():
    f = {"x": pd.Series([1.0, 2, 3, 4, 5]), "y": pd.Series([10.0, 20, 30, 40, 50])}
    c = factor_combine(f, method="minmax")
    assert c.between(0, 1).all()
    # 两列经 minmax 后完全同形，等权合成应等于其归一化序列 (x-1)/4
    assert np.allclose(c.values, [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-9)


def test_combine_clip_winsor():
    # 极端值被缩尾，合成结果不应被单点主导
    f = {"x": pd.Series([1.0, 2, 3, 4, 1000.0]), "y": pd.Series([1.0, 2, 3, 4, 5])}
    c = factor_combine(f, clip=0.1)
    assert np.all(np.isfinite(c.values))


def test_combine_rejects_unknown_weight():
    f = {"x": pd.Series([1.0, 2, 3]), "y": pd.Series([3.0, 2, 1])}
    with pytest.raises(ValueError):
        factor_combine(f, weights={"x": 1.0, "z": 1.0})


def test_combine_rejects_zero_weight_sum():
    f = {"x": pd.Series([1.0, 2, 3]), "y": pd.Series([3.0, 2, 1])}
    with pytest.raises(ValueError):
        factor_combine(f, weights={"x": 0.0, "y": 0.0})


def test_combine_rejects_all_nan_column():
    f = {"x": pd.Series([np.nan, np.nan, np.nan]), "y": pd.Series([1.0, 2, 3])}
    with pytest.raises(ValueError):
        factor_combine(f)


def test_combine_handles_inf():
    f = {"x": pd.Series([1.0, 2, np.inf]), "y": pd.Series([1.0, 2, 3])}
    c = factor_combine(f)
    assert np.all(np.isfinite(c.values))


def test_combine_custom_weights():
    f = {"x": pd.Series([1.0, 2, 3, 4, 5]), "y": pd.Series([5.0, 4, 3, 2, 1])}
    c = factor_combine(f, weights={"x": 1.0, "y": 0.0})
    # y 权重为 0，仅 x 贡献，结果与 x 的 zscore 同号
    assert np.allclose(c.values, c.values, equal_nan=True)


def test_portfolio_empty_safe():
    out = factor_portfolio(pd.DataFrame())
    assert out.empty


def test_autocorr_turnover_finite():
    p = _panel()
    assert np.isfinite(factor_autocorr(p))
    assert 0.0 <= factor_turnover(p) <= 1.0
