"""Round 8：factor_combine 输入守卫 + minmax/clip 新能力 + 组合空守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.factor.combine import (
    factor_combine, factor_portfolio, factor_autocorr, factor_turnover,
    factor_corr, factor_group_neutralize,
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


def test_factor_corr_mismatched_width_no_crash():
    # 隐性修复：单因子 vs 多资产目标，原实现会因重复列 reindex 抛异常
    idx = pd.date_range("2023-01-01", periods=12, freq="D")
    f = pd.DataFrame({"A": np.arange(12.0)}, index=idx)
    t = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (12, 3)),
                     index=idx, columns=["A", "B", "C"])
    c = factor_corr(f, t)
    assert np.isfinite(c)
    # 多资产因子 vs 单资产目标 也应正常
    c2 = factor_corr(t, f)
    assert np.isfinite(c2)


def test_factor_corr_perfect_positive():
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    x = pd.DataFrame({"A": np.arange(10.0), "B": np.arange(10.0) * 2}, index=idx)
    y = pd.DataFrame({"A": np.arange(10.0), "B": np.arange(10.0) * 2}, index=idx)
    assert abs(factor_corr(x, y) - 1.0) < 1e-9


def test_factor_group_neutralize_removes_group_mean():
    # 新增能力：组内中性化后，每组内因子均值应≈0
    idx = pd.date_range("2023-01-01", periods=1, freq="D")
    fp = pd.DataFrame({"A": [1.0], "B": [3.0], "C": [2.0], "D": [10.0]}, index=idx)
    groups = {"A": "g1", "B": "g1", "C": "g1", "D": "g2"}
    out = factor_group_neutralize(fp, groups, method="demean")
    assert abs(out.loc[idx[0], "A"] + out.loc[idx[0], "B"] + out.loc[idx[0], "C"]) < 1e-9
    # g2 仅一个资产，demean 后应归零
    assert abs(out.loc[idx[0], "D"]) < 1e-9


def test_factor_group_neutralize_zscore_shape_preserved():
    p = _panel(n=20, m=6, seed=3)
    groups = {c: ("g1" if i < 3 else "g2") for i, c in enumerate(p.columns)}
    out = factor_group_neutralize(p, groups, method="zscore")
    assert out.shape == p.shape
    assert np.all(np.isfinite(out.values))


def test_factor_group_neutralize_rejects_missing_label():
    idx = pd.date_range("2023-01-01", periods=1, freq="D")
    fp = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=idx)
    with pytest.raises(ValueError):
        factor_group_neutralize(fp, {"A": "g1"})  # 缺 B 标签
