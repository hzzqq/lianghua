"""风险成本/预算/分解（risk）：印花税语义修复 + ERC 零方差守卫 + Beta 无重叠 NaN 守卫。

续做轮 58（risk 集群，覆盖 cost/budget/decomp 此前未打磨模块）：
- 隐性语义修复：cost.CostModel 对 COVER(平空买入) 误征印花税，已改为仅 SELL/SHORT 征收。
- 隐性修复：budget._risk_contrib 在组合方差为 0 时除零爆 inf，现用 1e-12 兜底 + cov 方阵校验。
- 隐性修复：decomp.portfolio_beta / conditional_beta / marginal_var / incremental_var
  在基准无重叠或收益含 NaN 时原产出 NaN，现安全降级为 0.0/有限值。
- 新需求：cost.CostModel.breakdown 透明拆解每笔交易各项摩擦成本。
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.risk import cost as cost_mod
from lianghua.risk import budget
from lianghua.risk import decomp


def test_cover_not_taxed_but_sell_is():
    cm = cost_mod.CostModel(stamp_tax=0.001)
    # 平空(COVER)是买入，不应缴印花税
    assert cm.cost_for("COVER", 10.0, 100) == cm.cost_for("BUY", 10.0, 100)
    # 卖出应缴印花税
    sell = cm.cost_for("SELL", 10.0, 100)
    buy = cm.cost_for("BUY", 10.0, 100)
    assert sell > buy
    bd = cm.breakdown("SELL", 10.0, 100)
    assert bd["stamp_tax"] > 0 and bd["total"] >= bd["stamp_tax"]


def test_breakdown_components_sum():
    cm = cost_mod.STOCK_COST
    bd = cm.breakdown("BUY", 10.0, 1000)
    manual = bd["slippage"] + bd["commission"] + bd["impact"] + bd["stamp_tax"]
    assert abs(bd["total"] - max(manual, cm.min_fee)) < 1e-6


def test_erc_zero_variance_no_inf():
    # 零方差协方差（完全对冲/无波动）不应爆 inf
    cov = np.zeros((3, 3))
    w = budget.equal_risk_contribution(cov)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-9


def test_budget_rejects_bad_cov():
    with pytest.raises(ValueError):
        budget.equal_risk_contribution([[1, 0], [0, 1], [0, 0]])  # 非方阵
    with pytest.raises(ValueError):
        budget.equal_risk_contribution([[1.0, np.nan], [np.nan, 1.0]])


def test_erc_concentrates_risk_evenly():
    cov = np.array([[0.04, 0.0, 0.0],
                    [0.0, 0.04, 0.0],
                    [0.0, 0.0, 0.04]])
    w = budget.equal_risk_contribution(cov)
    rc = budget._risk_contrib(w, cov)
    assert np.allclose(rc, rc.mean(), atol=1e-3)


def test_decomp_beta_no_overlap_returns_zero():
    ret = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.01, 0.02]})
    # 基准日期完全错位 -> 无重叠，应安全返回 0.0 而非 NaN
    market = pd.Series([0.01, 0.02, 0.0], index=pd.date_range("2025-01-01", periods=3))
    beta = decomp.portfolio_beta(ret, [0.5, 0.5], market)
    assert beta == 0.0
    # conditional_beta 同样不应为 NaN
    cb = decomp.conditional_beta(ret, [0.5, 0.5], market)
    assert np.isfinite(cb)


def test_marginal_var_nan_cov_safe():
    ret = pd.DataFrame({"A": [0.01, np.nan, 0.02], "B": [0.0, 0.01, np.nan]})
    mv = decomp.marginal_var(ret, [0.5, 0.5])
    assert mv.notna().all()
