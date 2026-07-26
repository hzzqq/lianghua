"""CPPI 打磨测试：输入守卫 + 隐性 bug 修复 + 新增能力。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.portfolio.cppi import cppi, cppi_summary, _validate_cppi


def _ok_returns(n=120, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


def test_basic_shape_and_columns():
    r = _ok_returns()
    df = cppi(r)
    assert list(df.columns) == ["equity", "cushion", "risky_weight"]
    assert len(df) == len(r)
    assert np.all(df["risky_weight"].between(-1e-9, 1 + 1e-9))


def test_finite_no_nan_propagation():
    r = _ok_returns()
    df = cppi(r)
    assert np.all(np.isfinite(df["equity"].values))


def test_floor_pct_out_of_range_raises():
    r = _ok_returns(20)
    with pytest.raises(ValueError):
        cppi(r, floor_pct=1.2)
    with pytest.raises(ValueError):
        cppi(r, floor_pct=-0.1)


def test_multiplier_nonpositive_raises():
    r = _ok_returns(20)
    with pytest.raises(ValueError):
        cppi(r, multiplier=0.0)
    with pytest.raises(ValueError):
        cppi(r, multiplier=-2.0)


def test_init_cash_positive_raises():
    r = _ok_returns(20)
    with pytest.raises(ValueError):
        cppi(r, init_cash=0.0)


def test_nonfinite_returns_raises():
    r = _ok_returns(20)
    bad = r.copy()
    bad.iloc[5] = np.nan
    with pytest.raises(ValueError):
        cppi(bad)
    bad2 = r.copy()
    bad2.iloc[5] = np.inf
    with pytest.raises(ValueError):
        cppi(bad2)


def test_transaction_cost_reduces_final_equity():
    r = _ok_returns(200, seed=3)
    base = cppi(r, transaction_cost=0.0)
    costed = cppi(r, transaction_cost=0.001)
    assert costed["equity"].iloc[-1] <= base["equity"].iloc[-1] + 1e-6


def test_cppi_summary_sane():
    r = _ok_returns(150, seed=7)
    df = cppi(r, transaction_cost=0.0005)
    s = cppi_summary(df)
    assert set(["final_equity", "max_drawdown", "avg_risky_weight", "floor_breaches"]) <= set(s)
    assert np.isfinite(s["final_equity"])
    assert 0.0 <= s["max_drawdown"] <= 1.0
    assert s["avg_risky_weight"] >= -1e-9


def test_summary_rejects_bad_input():
    with pytest.raises(ValueError):
        cppi_summary(pd.DataFrame())
    with pytest.raises(ValueError):
        cppi_summary(pd.DataFrame({"equity": [1.0, np.nan]}))
