"""绩效指标模块测试：鲁棒守卫 + 新增 Calmar/水下曲线。"""
import math

import numpy as np
import pandas as pd
import pytest

from lianghua.perf import metrics


def _equity(values):
    return pd.Series(float(v) for v in values)


def test_sharpe_uptrend_positive():
    rng = np.random.RandomState(0)
    eq = _equity(100 + np.cumsum(rng.randn(300) + 0.05))  # 微涨
    assert metrics.sharpe(eq) > 0


def test_sharpe_flat_is_zero():
    eq = _equity([100.0] * 50)
    assert metrics.sharpe(eq) == 0.0


def test_sharpe_period_guard():
    with pytest.raises(ValueError):
        metrics.sharpe(_equity(np.arange(1, 51)), periods=0)


def test_max_drawdown_negative():
    eq = _equity([1.0, 1.1, 0.9, 1.2, 0.8])
    assert -1 < metrics.max_drawdown(eq) <= 0


def test_underwater_bounds():
    eq = _equity([1.0, 1.1, 0.9, 1.2, 0.8])
    uw = metrics.underwater(eq)
    assert ((uw >= -1) & (uw <= 0)).all()


def test_annual_return_first_zero_raises():
    with pytest.raises(ValueError):
        metrics.annual_return(_equity([0.0, 1.0, 2.0]))


def test_summary_has_calmar():
    eq = _equity([1.0, 1.1, 0.9, 1.2, 1.05])
    s = metrics.summary(eq, trades=[])
    assert "calmar" in s
    assert math.isfinite(s["calmar"])
    assert math.isfinite(s["sharpe"])
    assert math.isfinite(s["max_drawdown"])


def test_calmar_formula():
    eq = _equity([1.0, 1.1, 0.9, 1.2, 0.8])
    mdd = metrics.max_drawdown(eq)
    ar = metrics.annual_return(eq)
    assert abs(metrics.calmar_ratio(eq) - ar / abs(mdd)) < 1e-9


def test_equity_nan_raises():
    eq = pd.Series([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError):
        metrics.sharpe(eq)


def test_equity_empty_raises():
    with pytest.raises(ValueError):
        metrics.sharpe(pd.Series([], dtype=float))


def test_equity_non_series_raises():
    with pytest.raises((TypeError, ValueError)):
        metrics.sharpe(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_buyhold_missing_col_raises():
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(KeyError):
        metrics.buyhold_equity(df)


def test_attribution_two_assets():
    eqs = {
        "A": _equity([1.0, 1.2, 1.1, 1.3]),
        "B": _equity([1.0, 0.9, 1.0, 1.05]),
    }
    df = metrics.attribution(eqs)
    assert len(df) == 2
    assert "contribution" in df.columns
    assert "weight" in df.columns


def test_attribution_empty():
    df = metrics.attribution({})
    assert df.empty
