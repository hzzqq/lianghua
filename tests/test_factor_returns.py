"""factor.returns 测试：分组收益、IC、长度/inf 守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.factor.returns import (
    factor_long_short, factor_group_returns, factor_ic,
)


def _mk(n=500, seed=0, ic=0.6):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    r = ic * f + noise  # 前瞻收益与因子正相关
    return pd.Series(f), pd.Series(r)


def test_factor_long_short_positive_and_series():
    f, r = _mk(ic=0.8, seed=1)
    ls = factor_long_short(f, r)
    assert isinstance(ls, pd.Series)
    assert ls.iloc[0] > 0  # 因子与收益正相关 -> 多空为正


def test_factor_group_returns_monotonic():
    f, r = _mk(ic=0.8, seed=2)
    g = factor_group_returns(f, r, n_group=5)
    assert "long_short" in g.index
    assert g["long_short"] > 0
    # 高分组平均收益应高于低分组
    assert g.iloc[-2] > g.iloc[0]


def test_factor_ic_high():
    f, r = _mk(ic=0.8, seed=3)
    ic = factor_ic(f, r)
    assert isinstance(ic, float)
    assert ic > 0.3


def test_length_mismatch_raises():
    # 隐性修复验证：长度不一致必须显式报错而非静默截断
    f = pd.Series(np.arange(100.0))
    r = pd.Series(np.arange(80.0))
    with pytest.raises(ValueError):
        factor_long_short(f, r)
    with pytest.raises(ValueError):
        factor_ic(f, r)


def test_inf_guard():
    f = pd.Series([1.0, 2.0, np.inf, 4.0, 5.0])
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    # inf 被过滤后仍可计算（不崩溃、不返回 NaN 陷阱）
    ls = factor_long_short(f, r)
    assert np.isfinite(ls.iloc[0])
    ic = factor_ic(f, r)
    assert np.isfinite(ic)


def test_too_few_obs_raises():
    f = pd.Series([1.0])
    r = pd.Series([0.01])
    with pytest.raises(ValueError):
        factor_long_short(f, r)
