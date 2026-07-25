"""风险预算配置：等风险贡献（ERC）/ 目标风险预算。

在 portfolio.optimizer 的风险平价基础上，额外支持『指定风险预算』与
更稳健的迭代求解，便于风控约束落地。
"""
from __future__ import annotations

import numpy as np


def _risk_contrib(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    w = np.array(weights, float)
    c = np.array(cov, float)
    port_var = float(w @ c @ w)
    mrc = c @ w
    return w * mrc / port_var


def equal_risk_contribution(cov: np.ndarray, max_iter: int = 300,
                            tol: float = 1e-9) -> np.ndarray:
    """等风险贡献权重：每个资产对组合风险的贡献相同。"""
    n = len(cov)
    w = np.ones(n) / n
    for _ in range(max_iter):
        rc = _risk_contrib(w, cov)
        target = rc.mean()
        w = w * (target / (rc + 1e-12))
        w = w / w.sum()
        if np.max(np.abs(_risk_contrib(w, cov) - _risk_contrib(w, cov).mean())) < tol:
            break
    return w


def target_risk_budget(cov: np.ndarray, budgets: np.ndarray,
                       max_iter: int = 300) -> np.ndarray:
    """目标风险预算：各资产风险贡献比例 = budgets。"""
    budgets = np.array(budgets, float)
    budgets = budgets / budgets.sum()
    n = len(cov)
    w = np.ones(n) / n
    for _ in range(max_iter):
        rc = _risk_contrib(w, cov)
        w = w * (budgets / (rc + 1e-12))
        w = w / w.sum()
    return w
