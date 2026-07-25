"""risk.beta 测试：CAPM 基本量、滚动 Beta、inf 守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.risk.beta import beta, alpha_annual, capm_residuals, rolling_beta


def _mk(n=300, seed=0, beta_true=1.2):
    rng = np.random.default_rng(seed)
    bench = pd.Series(rng.normal(0.0005, 0.01, n), name="bench")
    asset = beta_true * bench + rng.normal(0, 0.005, n)
    return asset, bench


def test_beta_estimate():
    asset, bench = _mk(beta_true=1.5, seed=1)
    b = beta(asset, bench)
    assert np.isfinite(b)
    assert 1.2 < b < 1.8


def test_alpha_annual_near_zero():
    # 纯 beta 暴露、无超额收益 -> alpha 应接近 0
    asset, bench = _mk(beta_true=1.0, seed=2)
    a = alpha_annual(asset, bench)
    # 纯 beta 暴露、无真实 alpha：估计值应远小于资产自身年化波动量级（噪声约 0.07）
    assert abs(a) < 0.15


def test_capm_residuals_mean_small():
    asset, bench = _mk(beta_true=0.8, seed=3)
    res = capm_residuals(asset, bench)
    assert len(res) == len(asset)
    assert abs(res.mean()) < 0.002


def test_inf_guard():
    # 隐性修复验证：含 inf 的收益率不得让 beta 污染为 inf/nan
    asset = pd.Series([0.01, -0.02, np.inf, 0.005, -0.01])
    bench = pd.Series([0.005, -0.01, 0.02, 0.003, -0.008])
    b = beta(asset, bench)
    assert np.isfinite(b)


def test_rolling_beta_shape_and_window():
    asset, bench = _mk(n=120, seed=4)
    rb = rolling_beta(asset, bench, window=30, min_periods=10)
    assert len(rb) == len(asset)
    # 前 9 个窗口样本不足 -> NaN
    assert rb.iloc[:9].isna().all()
    # 后半段应给出有限 Beta 且在合理范围
    tail = rb.iloc[30:]
    assert tail.dropna().apply(lambda v: 0.5 < v < 2.0).all()


def test_rolling_beta_bad_window():
    asset, bench = _mk(seed=5)
    with pytest.raises(ValueError):
        rolling_beta(asset, bench, window=0)
