"""Round 9：portfolio.optimizer 风险贡献诊断 + max_weight 上限 + 输入守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.portfolio.optimizer import (
    equal_weight, inverse_vol, risk_parity, mean_variance, kelly,
    risk_contributions, optimize, portfolio_returns,
)


def _rets(seed=0, n=120):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (n, 3)),
        columns=["A", "B", "C"],
    )


def test_equal_weight_sums_to_one():
    w = equal_weight(_rets())
    assert np.isclose(w.sum(), 1.0)
    assert (w == 1 / 3).all()


def test_inverse_vol_weights_positive():
    w = inverse_vol(_rets())
    assert (w > 0).all() and np.isclose(w.sum(), 1.0)


def test_risk_parity_balances_contributions():
    w = risk_parity(_rets(seed=3))
    rc = risk_contributions(w, _rets(seed=3))
    # 风险平价下各资产贡献应接近相等（最大/最小 < 2）
    assert rc.max() / rc.min() < 2.0


def test_mean_variance_kelly_finite():
    for fn in (mean_variance, kelly):
        w = fn(_rets(seed=5))
        assert np.all(np.isfinite(w.values))
        assert np.isclose(w.sum(), 1.0)


def test_risk_contributions_sum_to_one():
    w = equal_weight(_rets())
    rc = risk_contributions(w, _rets())
    assert np.isclose(rc.sum(), 1.0, atol=1e-9)


def test_optimize_max_weight_cap():
    w = optimize(_rets(seed=2), method="equal", max_weight=0.4)
    assert (w <= 0.4 + 1e-9).all()
    assert np.isclose(w.sum(), 1.0)


def test_optimize_max_weight_risk_parity():
    w = optimize(_rets(seed=4), method="risk_parity", max_weight=0.5)
    assert (w <= 0.5 + 1e-9).all()


def test_optimize_unknown_method():
    with pytest.raises(ValueError):
        optimize(_rets(), method="nope")


def test_optimize_bad_max_weight():
    with pytest.raises(ValueError):
        optimize(_rets(), max_weight=1.5)


def test_inverse_vol_constant_column_no_nan():
    df = pd.DataFrame({"A": [0.01] * 50, "B": np.random.default_rng(1).normal(0, 0.01, 50)})
    w = inverse_vol(df)
    assert np.all(np.isfinite(w.values)) and np.isclose(w.sum(), 1.0)


def test_rejects_insufficient_rows():
    df = pd.DataFrame({"A": [0.01], "B": [0.02]})
    with pytest.raises(ValueError):
        risk_parity(df)
    with pytest.raises(ValueError):
        inverse_vol(df)


def test_rejects_empty_returns():
    with pytest.raises(ValueError):
        equal_weight(pd.DataFrame())


def test_rejects_non_dataframe():
    with pytest.raises(TypeError):
        equal_weight([1, 2, 3])


def test_portfolio_returns_shape():
    w = equal_weight(_rets())
    pr = portfolio_returns(w, _rets())
    assert len(pr) == 120
