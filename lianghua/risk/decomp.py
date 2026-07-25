"""风险分解（迭代 167–172, 178–180）。

把组合风险按资产/与基准关系分解。纯 pandas/numpy，零额外依赖。
依赖 tail._norm_ppf(零依赖正态分位) 与 tail.risk_contribution。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tail import _norm_ppf, risk_contribution


def marginal_var(returns: pd.DataFrame, weights, alpha: float = 0.05) -> pd.Series:
    """边际 VaR：组合 VaR 对各资产权重的偏导数（近似），∑w·MVaR=组合 VaR。"""
    w = np.asarray(weights, dtype=float)
    cov = returns.cov().values
    port_vol = np.sqrt(max(float(w @ cov @ w), 1e-12))
    mctr = (cov @ w) / port_vol
    z = _norm_ppf(alpha)
    return pd.Series(-z * mctr, index=returns.columns)


def incremental_var(returns: pd.DataFrame, weights, alpha: float = 0.05,
                    eps: float = 1e-4) -> pd.Series:
    """增量 VaR：把每个资产权重微调 eps 后组合 VaR 的变化（一阶近似）。"""
    w = np.asarray(weights, dtype=float)
    cov = returns.cov().values
    port_vol = np.sqrt(max(float(w @ cov @ w), 1e-12))
    z = _norm_ppf(alpha)
    base_var = -z * port_vol
    n = len(w)
    inc = np.zeros(n)
    for i in range(n):
        w2 = w.copy()
        w2[i] += eps
        v2 = np.sqrt(max(float(w2 @ cov @ w2), 1e-12))
        inc[i] = (-z * v2 - base_var) / eps
    return pd.Series(inc, index=returns.columns)


def diversification_ratio(returns: pd.DataFrame) -> float:
    """分散化比率：各资产等权加权波动之和 / 等权组合波动，>1 表示有分散收益。"""
    cov = returns.cov().values
    n = cov.shape[0]
    if n == 0:
        return 1.0
    w = np.ones(n) / n
    port_vol = np.sqrt(max(float(w @ cov @ w), 1e-12))
    mvol = np.sqrt(np.diag(cov))
    wavg = float(np.dot(w, mvol))
    return float(wavg / (port_vol + 1e-12))


def portfolio_beta(returns: pd.DataFrame, weights, market) -> float:
    """组合 Beta：组合收益对基准收益的敏感度。market 为收益序列（按日期对齐）。"""
    w = np.asarray(weights, dtype=float)
    R = returns.astype(float)
    m = pd.Series(market).astype(float).reindex(R.index)
    port = R.values @ w
    pr = pd.Series(port, index=R.index)
    idx = pr.index.intersection(m.index)
    pr, m = pr.loc[idx], m.loc[idx]
    if len(pr) < 2:
        return 0.0
    cov = np.cov(pr.values, m.values)
    return float(cov[0, 1] / (cov[1, 1] + 1e-12))


def systematic_var(returns: pd.DataFrame, weights, market, alpha: float = 0.05) -> float:
    """系统性 VaR：由 Beta 与基准波动推算的（不可分散）风险部分。"""
    beta = portfolio_beta(returns, weights, market)
    m = pd.Series(market).astype(float)
    mvol = m.std(ddof=0)
    z = _norm_ppf(alpha)
    return float(-z * beta * mvol)


def idiosyncratic_var(returns: pd.DataFrame, weights, market, alpha: float = 0.05) -> float:
    """特质 VaR：组合总 VaR − 系统性 VaR（可分散的个股特异风险部分）。"""
    w = np.asarray(weights, dtype=float)
    cov = returns.cov().values
    port_vol = np.sqrt(max(float(w @ cov @ w), 1e-12))
    z = _norm_ppf(alpha)
    total = -z * port_vol
    sys = systematic_var(returns, weights, market, alpha)
    return float(total - sys)


def conditional_beta(returns: pd.DataFrame, weights, market, alpha: float = 0.05) -> float:
    """条件 Beta：仅当基准处于左尾(≤α分位)时组合对基准的敏感度（崩盘β）。"""
    R = returns.astype(float)
    m = pd.Series(market).astype(float).reindex(R.index)
    port = R.values @ np.asarray(weights, dtype=float)
    pr = pd.Series(port, index=R.index)
    idx = pr.index.intersection(m.index)
    pr, m = pr.loc[idx], m.loc[idx]
    if len(pr) < 5:
        return portfolio_beta(returns, weights, market)
    tail = m <= m.quantile(alpha)
    if tail.sum() < 5:
        return portfolio_beta(returns, weights, market)
    cov = np.cov(pr[tail].values, m[tail].values)
    return float(cov[0, 1] / (cov[1, 1] + 1e-12))


def risk_parity_deviation(returns: pd.DataFrame, weights) -> float:
    """风险平价偏离度：各资产风险贡献与等权目标的欧氏距离，0=完美风险平价。"""
    rc = risk_contribution(returns, weights)
    rc = rc / (rc.sum() + 1e-12)
    n = len(rc)
    target = 1.0 / n
    return float(np.sqrt(((rc - target) ** 2).sum()))


def concentration_index(weights) -> float:
    """集中度指数(HHI)：权重平方和，越接近 1 越集中。"""
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s == 0:
        return 0.0
    w = w / s
    return float(np.sum(w ** 2))
