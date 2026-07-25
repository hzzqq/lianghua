"""尾部与回撤风险（迭代 129）：
component_var / cdar(条件回撤风险) / downside_deviation。

纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """下行标准差：只统计低于最小可接受收益(MAR)的波动。"""
    r = pd.Series(returns).astype(float).dropna()
    downside = np.minimum(r - mar, 0.0)
    return float(np.sqrt((downside ** 2).mean()))


def _norm_ppf(p: float) -> float:
    """标准正态分位点（Acklam 有理逼近，零依赖）。"""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def component_var(returns: pd.DataFrame, weights, alpha: float = 0.05) -> pd.Series:
    """成分 VaR：把组合 VaR 按各资产的边际贡献分解，和≈组合 VaR。

    参数法（正态近似）：成分风险贡献 CCTR_i = w_i·(Σw)_i/σ_p，
    再乘 -z_α 转为 VaR 口径（正数=风险贡献）。∑CCTR = σ_p。
    """
    w = np.asarray(weights, dtype=float)
    cov = returns.cov().values
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(port_var) + 1e-12
    mctr = (cov @ w) / port_vol           # 边际风险贡献
    cctr = w * mctr                        # 成分风险贡献（和=port_vol）
    z = _norm_ppf(alpha)
    comp = -z * cctr                       # 转为 VaR 口径（正数=风险）
    return pd.Series(comp, index=returns.columns)


def cdar(equity: pd.Series, alpha: float = 0.05) -> float:
    """条件回撤风险(CDaR)：最差 alpha 分位回撤的平均值。"""
    e = pd.Series(equity).astype(float)
    peak = e.cummax()
    dd = 1.0 - e / peak                    # 回撤为正数
    if len(dd) == 0:
        return 0.0
    thr = np.quantile(dd, 1 - alpha)
    tail = dd[dd >= thr]
    return float(tail.mean() if len(tail) else thr)


def tail_dependence(ra: pd.Series, rb: pd.Series, alpha: float = 0.05) -> float:
    """经验尾部依赖系数（左尾）：P(rb ≤ q_b) 在 ra ≤ q_a 条件下的概率。

    取值 ∈[0,1]；接近 0 表示尾部独立，接近 1 表示危机时同跌（系统性风险）。
    """
    a = pd.Series(ra).astype(float).dropna()
    b = pd.Series(rb).astype(float).dropna()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    if len(a) < 20:
        return 0.0
    qa = a.quantile(alpha)
    qb = b.quantile(alpha)
    in_a = a <= qa
    if not in_a.any():
        return 0.0
    return float((in_a & (b <= qb)).sum() / in_a.sum())


def risk_contribution(returns: pd.DataFrame, weights, annual: int = 252) -> pd.Series:
    """成分风险贡献（波动口径）：CCTR_i = w_i·(Σw)_i / σ_p，Σ 年化。

    ∑CCTR = 年化组合波动 σ_p；正值表示该资产增加组合风险。
    weights 可为 Series/ndarray，按 returns.columns 对齐。
    """
    w = np.asarray(weights, dtype=float)
    cov = returns.cov().values * annual
    port_vol = np.sqrt(max(float(w @ cov @ w), 1e-12))
    mctr = (cov @ w) / port_vol
    cctr = w * mctr
    return pd.Series(cctr, index=returns.columns)
