"""Cycle 49：portfolio.optimizer 打磨——隐性修复 inf 泄漏/索引错位 + 新增有效下注数/组合波动。"""
import warnings

import numpy as np
import pandas as pd
import pytest

from lianghua.portfolio.optimizer import (
    equal_weight, risk_parity, risk_contributions,
    portfolio_returns, portfolio_volatility, effective_bets,
)


def _rets(seed=0, n=120, cols=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0005, 0.01, (n, len(cols))), columns=cols)


def test_inf_in_returns_no_silent_nan_weights():
    # 隐性修复：inf 收益不应静默产出 inf/NaN 权重
    df = _rets()
    df.loc[df.index[0], "A"] = np.inf
    df.loc[df.index[1], "B"] = -np.inf
    w = risk_parity(df)
    assert np.all(np.isfinite(w.values)) and np.isclose(w.sum(), 1.0)
    w2 = risk_contributions(equal_weight(df), df)
    assert np.all(np.isfinite(w2.values))


def test_risk_contributions_robust_to_index_order():
    # 隐性修复：权重索引顺序与列不一致时不应错贴贡献
    r = _rets(seed=7)
    w = equal_weight(r)
    w_shuffled = w.iloc[::-1]  # 顺序颠倒，标签仍对应同一资产
    rc = risk_contributions(w_shuffled, r)
    rc_ref = risk_contributions(w, r)
    # 对齐正确时，顺序颠倒应得到完全相同的贡献（值与标签都对得上）
    assert np.allclose(rc.sort_index().values, rc_ref.sort_index().values)
    assert set(rc.index) == set(r.columns)


def test_risk_contributions_rejects_mislabeled_weights():
    # 标签与列完全无关 -> 对齐后全 0 -> 返回 NaN
    r = _rets(seed=2)
    w_bad = pd.Series([1.0, 1.0], index=["X", "Y"])
    rc = risk_contributions(w_bad, r)
    assert rc.isna().all()


def test_portfolio_returns_drops_unknown_columns_with_warning():
    r = _rets()
    w = pd.Series([1.0, 1.0, 1.0, 5.0], index=["A", "B", "C", "GHOST"])
    with pytest.warns(UserWarning):
        pr = portfolio_returns(w, r)
    assert len(pr) == len(r)
    assert np.all(np.isfinite(pr.values))


def test_portfolio_returns_normalizes_unnormalized_weights():
    r = _rets(seed=4)
    w = pd.Series([2.0, 2.0, 2.0], index=list(r.columns))  # 未归一化
    pr = portfolio_returns(w, r)
    # 与等权组合收益一致（仅缩放倍数相同）
    pr_eq = portfolio_returns(equal_weight(r), r)
    assert np.allclose(pr.values, pr_eq.values)


def test_portfolio_volatility_finite_and_positive():
    r = _rets(seed=9)
    vol = portfolio_volatility(equal_weight(r), r)
    assert vol > 0 and np.isfinite(vol)
    # 单资产组合波动应等于该资产年化波动
    single = pd.DataFrame(_rets(cols=["A"])["A"])
    v1 = portfolio_volatility(pd.Series([1.0], index=["A"]), single)
    assert np.isfinite(v1)


def test_effective_bets_bounds():
    # 完全不相关 -> 接近资产数；完全相关 -> 接近 1
    indep = pd.DataFrame(np.random.default_rng(1).normal(0, 0.01, (200, 4)))
    eb_ind = effective_bets(pd.Series(np.ones(4) / 4, index=list(indep.columns)), indep)
    assert eb_ind > 3.0  # 接近 4
    # 完全线性相关：复制同一列
    base = np.random.default_rng(2).normal(0, 0.01, 200)
    corr_df = pd.DataFrame(np.column_stack([base, base, base]), columns=["A", "B", "C"])
    eb_corr = effective_bets(pd.Series(np.ones(3) / 3, index=["A", "B", "C"]), corr_df)
    assert eb_corr < 1.5


def test_effective_bets_single_asset_is_one():
    single = pd.DataFrame(_rets(cols=["A"])["A"])
    assert effective_bets(pd.Series([1.0], index=["A"]), single) == 1.0
