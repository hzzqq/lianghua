"""风险 VaR 测试（Round 18）：新增蒙特卡洛 VaR + 隐性守卫（conf/horizon/notional/方法）。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.risk.var import (
    historical_var, parametric_var, cvar, monte_carlo_var, var_report,
)


def _rets(n=500, seed=2):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


def test_historical_var_positive():
    r = _rets()
    v = historical_var(r, 0.95)
    assert v >= 0
    assert v < 0.2  # 合理量级


def test_cvar_ge_var():
    r = _rets()
    assert cvar(r, 0.95) >= historical_var(r, 0.95) - 1e-9


def test_monte_carlo_var_bootstrap():
    r = _rets()
    v = monte_carlo_var(r, 0.95, n_sims=2000, method="bootstrap", seed=7)
    assert v >= 0
    # bootstrap VaR 应接近历史法量级
    assert abs(v - historical_var(r, 0.95)) < 0.01


def test_monte_carlo_var_normal_reproducible():
    r = _rets()
    a = monte_carlo_var(r, 0.99, n_sims=1000, method="normal", seed=11)
    b = monte_carlo_var(r, 0.99, n_sims=1000, method="normal", seed=11)
    assert a == b  # 同 seed 可复现


def test_monte_carlo_bad_n_sims():
    with pytest.raises(ValueError):
        monte_carlo_var(_rets(), 0.95, n_sims=0)


def test_monte_carlo_bad_method():
    with pytest.raises(ValueError):
        monte_carlo_var(_rets(), 0.95, method="foo")


def test_z_score_conf_guard():
    with pytest.raises(ValueError):
        parametric_var(_rets(), conf=1.5)
    with pytest.raises(ValueError):
        cvar(_rets(), conf=-0.1)


def test_var_report_guards():
    r = _rets()
    with pytest.raises(ValueError):
        var_report(r, horizon_days=0)
    with pytest.raises(ValueError):
        var_report(r, notional=-100)
    with pytest.raises(ValueError):
        var_report(r, confs=(1.2,))


def test_var_report_scales_with_horizon():
    r = _rets()
    rep1 = var_report(r, confs=(0.95,), horizon_days=1)
    rep10 = var_report(r, confs=(0.95,), horizon_days=10)
    # 持有期越长 VaR 越大（sqrt 缩放）
    assert rep10["历史VaR"].iloc[0] > rep1["历史VaR"].iloc[0]


def test_var_report_notional_amount():
    r = _rets()
    rep = var_report(r, confs=(0.95,), notional=1_000_000)
    assert "历史VaR金额" in rep.columns
    assert rep["历史VaR金额"].iloc[0] == rep["历史VaR"].iloc[0] * 1_000_000
