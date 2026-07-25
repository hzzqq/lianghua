"""risk.var 测试：Cornish-Fisher 修正 VaR、inf 守卫、蒙特卡洛可复现。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.risk.var import (
    historical_var,
    parametric_var,
    cvar,
    monte_carlo_var,
    cornish_fisher_var,
    var_report,
)


def _normal(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


def test_cornish_fisher_basic():
    r = _normal()
    v = cornish_fisher_var(r, 0.95)
    assert np.isfinite(v)
    assert v >= 0.0
    # 近正态下，修正 VaR 应接近参数法 VaR（同 mu/sigma 口径）
    pv = parametric_var(r, 0.95)
    assert abs(v - pv) < 1e-3


def test_cornish_fisher_fatter_tail_larger():
    # 构造带厚尾（高峰度）的分布，修正 VaR 应大于正态参数法 VaR
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.01, 8000)
    tail = rng.standard_t(3, 2000) * 0.02  # 厚尾
    r = pd.Series(np.concatenate([base, tail]))
    cf = cornish_fisher_var(r, 0.99)
    pv = parametric_var(r, 0.99)
    assert cf > pv


def test_cornish_fisher_flat_series_zero():
    r = pd.Series(np.full(50, 0.001))
    assert cornish_fisher_var(r, 0.95) == 0.0


def test_clean_drops_inf():
    # 隐性修复验证：含 inf 的输入不应让分位数/摘要崩出 inf
    r = pd.Series([0.01, -0.02, np.inf, 0.005, -0.01, np.inf, -0.03])
    for fn in (historical_var, parametric_var, cvar, cornish_fisher_var):
        assert np.isfinite(fn(r, 0.95))


def test_clean_too_short_raises():
    with pytest.raises(ValueError):
        cornish_fisher_var(pd.Series([np.nan, np.nan]), 0.95)
    with pytest.raises(ValueError):
        historical_var(pd.Series([0.01]), 0.95)


def test_monte_carlo_reproducible():
    r = _normal()
    a = monte_carlo_var(r, 0.95, n_sims=2000, seed=42)
    b = monte_carlo_var(r, 0.95, n_sims=2000, seed=42)
    assert a == b
    assert np.isfinite(a) and a >= 0.0


def test_monte_carlo_methods():
    r = _normal()
    boot = monte_carlo_var(r, 0.95, n_sims=2000, method="bootstrap", seed=1)
    norm = monte_carlo_var(r, 0.95, n_sims=2000, method="normal", seed=1)
    assert boot >= 0.0 and norm >= 0.0
    with pytest.raises(ValueError):
        monte_carlo_var(r, 0.95, method="bad")


def test_cvar_ge_var():
    r = _normal(seed=3)
    assert cvar(r, 0.95) >= historical_var(r, 0.95) - 1e-12


def test_var_report_notional():
    r = _normal(seed=5)
    df = var_report(r, confs=(0.95, 0.99), notional=1_000_000)
    assert "CVaR金额" in df.columns
    assert (df["CVaR金额"] >= 0).all()
    # 高置信度 VaR 应不小于低置信度
    assert df["历史VaR"].iloc[1] >= df["历史VaR"].iloc[0]
