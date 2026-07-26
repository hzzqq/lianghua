"""风险预算配置：等风险贡献（ERC）/ 目标风险预算。

在 portfolio.optimizer 的风险平价基础上，额外支持『指定风险预算』与
更稳健的迭代求解，便于风控约束落地。
"""
from __future__ import annotations

import numpy as np


def _check_cov(cov) -> np.ndarray:
    a = np.asarray(cov, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("cov 必须为非空方阵")
    if not np.isfinite(a).all():
        raise ValueError("cov 含 NaN/inf，无法求解风险预算")
    return a


def _risk_contrib(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    w = np.array(weights, float)
    c = np.array(cov, float)
    # 隐性修复：组合方差为 0（零方差/完全对冲）时除零会爆出 inf，
    # 用 1e-12 兜底保证数值稳定（权重不发散）。
    port_var = max(float(w @ c @ w), 1e-12)
    mrc = c @ w
    return w * mrc / port_var


def equal_risk_contribution(cov: np.ndarray, max_iter: int = 300,
                            tol: float = 1e-9) -> np.ndarray:
    """等风险贡献权重：每个资产对组合风险的贡献相同。"""
    c = _check_cov(cov)
    n = c.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        rc = _risk_contrib(w, c)
        if not np.isfinite(rc).all() or float(np.abs(rc).sum()) < 1e-15:
            # 组合无风险可分配（零方差/完全对冲）-> 退化为等权，避免除零发散
            w = np.ones(n) / n
            break
        target = rc.mean()
        new_w = w * (target / (rc + 1e-12))
        new_w = new_w / new_w.sum()
        if not np.isfinite(new_w).all():
            break
        w = new_w
        if np.max(np.abs(_risk_contrib(w, c) - _risk_contrib(w, c).mean())) < tol:
            break
    return w


def target_risk_budget(cov: np.ndarray, budgets: np.ndarray,
                       max_iter: int = 300) -> np.ndarray:
    """目标风险预算：各资产风险贡献比例 = budgets。"""
    c = _check_cov(cov)
    budgets = np.array(budgets, float)
    budgets = budgets / budgets.sum()
    n = c.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        rc = _risk_contrib(w, c)
        if not np.isfinite(rc).all() or float(np.abs(rc).sum()) < 1e-15:
            w = np.ones(n) / n
            break
        new_w = w * (budgets / (rc + 1e-12))
        new_w = new_w / new_w.sum()
        if not np.isfinite(new_w).all():
            break
        w = new_w
    return w
