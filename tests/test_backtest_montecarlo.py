"""蒙特卡洛模拟：新增 simulate_paths 分位带能力 + 隐性 bug 修复回归测试。

覆盖：
- 种子可复现性（修复 np.random.seed 不生效的隐性非确定性 bug）
- block<=0 死锁守卫
- 空 / 非有限 / 非正输入守卫
- simulate_paths 分位带形状与确定性
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.backtest.montecarlo import bootstrap, parametric, simulate_paths


def _make_equity(n=200, seed=1, drift=0.0005, vol=0.01):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    eq = 1000.0 * np.cumprod(1.0 + rets)
    return pd.Series(eq, name="equity")


def test_bootstrap_is_deterministic_with_seed():
    eq = _make_equity()
    a = bootstrap(eq, n=200, seed=7)
    b = bootstrap(eq, n=200, seed=7)
    # 关键回归：seed 必须真正生效，两次结果应逐元素一致
    assert np.array_equal(a["returns"], b["returns"])
    assert a["mean_return"] == pytest.approx(b["mean_return"])


def test_parametric_is_deterministic_with_seed():
    eq = _make_equity()
    a = parametric(eq, n=200, seed=11)
    b = parametric(eq, n=200, seed=11)
    assert np.array_equal(a["returns"], b["returns"])


def test_different_seed_gives_different_paths():
    eq = _make_equity()
    a = bootstrap(eq, n=200, seed=1)
    b = bootstrap(eq, n=200, seed=2)
    assert not np.array_equal(a["returns"], b["returns"])


def test_block_le_zero_raises():
    eq = _make_equity()
    with pytest.raises(ValueError):
        bootstrap(eq, block=0)
    with pytest.raises(ValueError):
        bootstrap(eq, block=-3)


def test_n_le_zero_raises():
    eq = _make_equity()
    with pytest.raises(ValueError):
        bootstrap(eq, n=0)
    with pytest.raises(ValueError):
        parametric(eq, n=-1)


def test_empty_equity_raises():
    with pytest.raises(ValueError):
        bootstrap(pd.Series([], dtype=float))
    with pytest.raises(ValueError):
        bootstrap(pd.Series([100.0]))  # 少于 2 个观测点


def test_nonfinite_equity_raises():
    eq = _make_equity()
    bad = eq.copy()
    bad.iloc[5] = np.nan
    with pytest.raises(ValueError):
        bootstrap(bad)
    bad2 = eq.copy()
    bad2.iloc[5] = np.inf
    with pytest.raises(ValueError):
        parametric(bad2)


def test_nonpositive_init_cash_raises():
    eq = _make_equity()
    with pytest.raises(ValueError):
        bootstrap(eq, init_cash=0.0)
    with pytest.raises(ValueError):
        bootstrap(eq, init_cash=-5.0)


def test_summary_fields_finite_and_sane():
    eq = _make_equity()
    r = bootstrap(eq, n=300, seed=3)
    assert np.isfinite(r["mean_return"])
    assert 0.0 <= r["prob_loss"] <= 1.0
    assert r["best"] >= r["worst"]
    assert np.isfinite(r["mean_max_drawdown"])
    assert r["std_return"] >= 0.0


def test_simulate_paths_shape_and_keys():
    eq = _make_equity()
    res = simulate_paths(eq, n=150, seed=5)
    n_days = len(eq) - 1
    assert res["days"].shape == (n_days + 1,)
    assert res["median"].shape == (n_days + 1,)
    assert res["mean"].shape == (n_days + 1,)
    # 分位带形状与键齐全
    for p in (5, 25, 50, 75, 95):
        assert p in res["bands"]
        assert res["bands"][p].shape == (n_days + 1,)
    # p5 <= p50 <= p95 在终点处成立（大多数情况）
    end = -1
    assert res["bands"][5][end] <= res["bands"][95][end]
    assert 0.0 <= res["final_return"]["prob_loss"] <= 1.0


def test_simulate_paths_deterministic():
    eq = _make_equity()
    a = simulate_paths(eq, n=120, seed=9)
    b = simulate_paths(eq, n=120, seed=9)
    assert np.array_equal(a["median"], b["median"])


def test_simulate_paths_empty_percentiles_raises():
    eq = _make_equity()
    with pytest.raises(ValueError):
        simulate_paths(eq, percentiles=())
