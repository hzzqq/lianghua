"""补充风险函数（迭代 181+，超自驱动目标）：熵风险 / 集中度风险 / 相关性风险 /
期望损失(ES) / 流动性调整 VaR / 状态 VaR / 压力 VaR / 最大亏损概率。

纯 pandas/numpy，零额外依赖（严禁 scipy，正态分位走 risk/tail._norm_ppf）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tail import _norm_ppf


def entropy_risk(weights) -> float:
    """权重分布的信息熵（归一化 ∈[0,1]）：越低越集中、分散不足风险越高。"""
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    if len(w) <= 1:
        return 0.0
    p = w / w.sum()
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(len(p)))


def concentration_risk(weights) -> float:
    """集中度风险（Herfindahl 指数 = Σw²，∈[0,1]）：越高越集中于少数资产。"""
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s == 0:
        return 0.0
    p = w / s
    return float(np.sum(p ** 2))


def correlation_risk(returns) -> float:
    """相关性风险：资产间平均绝对相关系数（含自身则先去掉对角线）。
    越高说明同涨同跌、分散失效的风险越大。"""
    if isinstance(returns, pd.DataFrame):
        corr = returns.corr().values
    else:
        corr = np.corrcoef(np.asarray(returns, dtype=float), rowvar=False)
    n = corr.shape[0]
    if n < 2:
        return 0.0
    off = corr[~np.eye(n, dtype=bool)]
    return float(np.mean(np.abs(off)))


def expected_shortfall(returns, alpha: float = 0.05) -> float:
    """期望损失(ES/CVaR)：左尾 alpha 分位以内收益的均值（负值=损失）。"""
    r = pd.Series(returns).astype(float).dropna().values
    if len(r) == 0:
        return 0.0
    var = np.percentile(r, alpha * 100)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)


def liquidity_adjusted_var(returns, notional: float = 1_000_000.0,
                           alpha: float = 0.05, liquidity_haircut: float = 0.01) -> float:
    """流动性调整 VaR：历史 VaR（损失额）+ 流动性价差成本（按名义本金计）。"""
    r = pd.Series(returns).astype(float).dropna().values
    if len(r) == 0:
        return 0.0
    var_frac = -np.percentile(r, alpha * 100)
    return float(var_frac * notional + notional * liquidity_haircut)


def regime_var(returns, regime_mask, alpha: float = 0.05) -> float:
    """状态 VaR：仅在『风险状态』(regime_mask=True) 的子样本上算历史 VaR（损失额）。"""
    r = pd.Series(returns).astype(float)
    mask = pd.Series(regime_mask).reindex(r.index).fillna(False).astype(bool)
    sub = r[mask].dropna().values
    if len(sub) == 0:
        return 0.0
    return float(-np.percentile(sub, alpha * 100))


def stress_var(returns, shock: float = 0.20, alpha: float = 0.05) -> float:
    """压力 VaR：在施加 -shock 冲击情景后（所有收益下移 shock）的历史 VaR（损失额）。"""
    r = pd.Series(returns).astype(float).dropna().values
    if len(r) == 0:
        return 0.0
    stressed = r - abs(shock)
    return float(-np.percentile(stressed, alpha * 100))


def max_loss_prob(returns, threshold: float = -0.05) -> float:
    """最大亏损概率：单期收益低于 threshold 的比例（∈[0,1]）。"""
    r = pd.Series(returns).astype(float).dropna().values
    if len(r) == 0:
        return 0.0
    return float((r < threshold).mean())


__all__ = ["entropy_risk", "concentration_risk", "correlation_risk",
           "expected_shortfall", "liquidity_adjusted_var", "regime_var",
           "stress_var", "max_loss_prob"]
