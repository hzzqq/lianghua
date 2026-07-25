"""补充绩效比率（迭代 157–166）。

输入统一为净值曲线 equity(pd.Series) 或收益序列 returns(pd.Series)；
需基准的函数额外传入 benchmark 净值/收益。纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ratios import _returns, gain_to_pain


def _max_drawdown(equity: pd.Series) -> float:
    e = equity.astype(float)
    peak = e.cummax()
    return float((e / peak - 1.0).min())


def burke_ratio(equity: pd.Series, annual: int = 252) -> float:
    """Burke 比率：年化收益 / sqrt(各期回撤平方和)，比 Calmar 更惩罚深回撤。"""
    r = _returns(equity)
    if len(r) == 0:
        return 0.0
    ann_ret = (1 + r.mean()) ** annual - 1
    peak = equity.astype(float).cummax()
    dd = equity.astype(float) / peak - 1.0
    denom = np.sqrt((dd ** 2).sum() / (len(dd) + 1e-12))
    return float(ann_ret / (denom + 1e-12))


def common_sense_ratio(equity: pd.Series) -> float:
    """常识比率：收益痛苦比 / |最大回撤|，衡量单位风险补偿。"""
    gtp = gain_to_pain(equity)
    mdd = _max_drawdown(equity)
    return float(gtp / (-mdd + 1e-12))


def sterling_ratio(equity: pd.Series, risk_free: float = 0.0, annual: int = 252) -> float:
    """Sterling 比率：年化超额收益 / 平均回撤（比 Calmar 用平均回撤更稳）。"""
    r = _returns(equity)
    if len(r) == 0:
        return 0.0
    ann_ret = (1 + r.mean()) ** annual - 1 - risk_free
    peak = equity.astype(float).cummax()
    dd = equity.astype(float) / peak - 1.0
    avg_dd = dd.mean()
    return float(ann_ret / (-avg_dd + 1e-12))


def pain_index(equity: pd.Series) -> float:
    """痛苦指数：累计回撤 / 期数，衡量回撤的持久痛感。"""
    e = equity.astype(float)
    peak = e.cummax()
    dd = e / peak - 1.0
    if len(dd) == 0:
        return 0.0
    return float(dd.cumsum().iloc[-1] / len(dd))


def sharpe_penalized(equity: pd.Series, annual: int = 252) -> float:
    """惩罚夏普：对负偏/高峰度收益做折扣，抑制尖峰厚尾带来的虚高 Sharpe。"""
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    sharpe = r.mean() / (r.std(ddof=0) + 1e-12) * np.sqrt(annual)
    sk = r.skew()
    ku = r.kurt()
    pen = 1.0 + max(0.0, -sk) * 0.1 + max(0.0, ku - 3) * 0.05
    return float(sharpe / pen)


def return_skew(returns) -> float:
    """收益偏度：<0 左尾更肥（常伴随崩盘风险）。"""
    r = pd.Series(returns).astype(float).dropna()
    return float(r.skew()) if len(r) > 2 else 0.0


def return_kurtosis(returns) -> float:
    """收益峰度（超额峰度）：>0 厚尾，极端事件概率高于正态。"""
    r = pd.Series(returns).astype(float).dropna()
    return float(r.kurt()) if len(r) > 3 else 0.0


def _aligned(equity: pd.Series, benchmark) -> tuple:
    r = _returns(equity)
    b = pd.Series(benchmark).astype(float)
    b = b if (b > 1.5).mean() < 0.5 else b.pct_change()
    idx = r.index.intersection(b.index)
    return r.loc[idx], b.loc[idx]


def treynor_ratio(equity: pd.Series, benchmark, risk_free: float = 0.0,
                  annual: int = 252) -> float:
    """Treynor 比率：(年化超额收益) / 组合 Beta，衡量每单位系统性风险补偿。"""
    r, b = _aligned(equity, benchmark)
    if len(r) < 2:
        return 0.0
    ann_ret = (1 + r.mean()) ** annual - 1 - risk_free
    cov = np.cov(r.values, b.values)
    beta = cov[0, 1] / (cov[1, 1] + 1e-12)
    return float(ann_ret / (beta + 1e-12))


def jensen_alpha(equity: pd.Series, benchmark, risk_free: float = 0.0,
                 annual: int = 252) -> float:
    """Jensen's Alpha：CAPM 框架下的超额收益（正=跑赢市场基准）。"""
    r, b = _aligned(equity, benchmark)
    if len(r) < 3:
        return 0.0
    ann_r = (1 + r.mean()) ** annual - 1
    ann_b = (1 + b.mean()) ** annual - 1
    cov = np.cov(r.values, b.values)
    beta = cov[0, 1] / (cov[1, 1] + 1e-12)
    return float(ann_r - (risk_free + beta * (ann_b - risk_free)))


def capm_beta(equity: pd.Series, benchmark) -> float:
    """CAPM Beta：组合收益对基准收益的敏感度。"""
    r, b = _aligned(equity, benchmark)
    if len(r) < 2:
        return 0.0
    cov = np.cov(r.values, b.values)
    return float(cov[0, 1] / (cov[1, 1] + 1e-12))
