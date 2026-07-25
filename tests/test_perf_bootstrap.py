"""Bootstrap 置信区间：新增 ci 自定义/boot_ci/失败隔离 + 输入守卫回归。

覆盖：
- 非有限 / 少于 2 点 / n_boot<1 / ci 非法 守卫
- 逐样本失败隔离（metric 全失败 -> 明确 RuntimeError）
- 种子可复现（distribution 逐元素一致）
- bootstrap_ci 便捷封装与 alpha 守卫
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.perf.bootstrap import bootstrap_metric, bootstrap_ci
from lianghua.perf.annualize import annual_return


def _equity(n=300, seed=1):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.01, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100.0 * np.cumprod(1.0 + ret), index=idx)


def test_keys_and_order():
    eq = _equity()
    b = bootstrap_metric(eq, annual_return)
    for k in ("mean", "median", "std", "low_95", "high_95", "low", "high",
             "ci", "n", "n_valid", "n_fail"):
        assert k in b
    assert b["low_95"] <= b["high_95"]
    assert np.isfinite(b["mean"])
    assert b["n_valid"] == b["n"]
    assert b["n_fail"] == 0


def test_deterministic_with_seed():
    eq = _equity()
    a = bootstrap_metric(eq, annual_return, seed=42, return_dist=True)
    b = bootstrap_metric(eq, annual_return, seed=42, return_dist=True)
    assert np.array_equal(a["distribution"], b["distribution"])


def test_custom_ci():
    eq = _equity()
    res = bootstrap_metric(eq, annual_return, ci=(5, 95))
    assert res["ci"] == [5, 95]
    assert res["low"] <= res["high"]


def test_nonfinite_equity_raises():
    eq = _equity()
    bad = eq.copy()
    bad.iloc[10] = np.nan
    with pytest.raises(ValueError):
        bootstrap_metric(bad, annual_return)


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        bootstrap_metric(pd.Series([100.0]), annual_return)


def test_n_boot_guard():
    with pytest.raises(ValueError):
        bootstrap_metric(_equity(), annual_return, n_boot=0)


def test_invalid_ci_raises():
    with pytest.raises(ValueError):
        bootstrap_metric(_equity(), annual_return, ci=(10, 200))
    with pytest.raises(ValueError):
        bootstrap_metric(_equity(), annual_return, ci=(50, 40))


def test_all_metric_failures_raise():
    def boom(eq):
        raise ValueError("nope")
    with pytest.raises(RuntimeError):
        bootstrap_metric(_equity(), boom)


def test_bootstrap_ci_tuple():
    eq = _equity()
    lo, hi = bootstrap_ci(eq, annual_return, alpha=0.10, n_boot=200)
    assert lo <= hi
    # 同一结果内：分位单调 low_95(2.5%) <= low(5%) <= high(95%) <= high_95(97.5%)
    res = bootstrap_metric(eq, annual_return, n_boot=500, seed=0, ci=(5, 95))
    assert res["low_95"] <= res["low"] <= res["high"] <= res["high_95"]


def test_bootstrap_ci_alpha_guard():
    with pytest.raises(ValueError):
        bootstrap_ci(_equity(), annual_return, alpha=1.0)
