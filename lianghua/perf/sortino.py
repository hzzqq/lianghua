"""下行风险调整指标：Sortino / Calmar / Omega。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _returns(equity: pd.Series) -> pd.Series:
    eq = equity.astype(float)
    return eq.pct_change().dropna()


def sortino(equity: pd.Series, target: float = 0.0, periods: int = 252) -> float:
    """年化 Sortino 比率。"""
    r = _returns(equity)
    if r.empty:
        return 0.0
    downside = r[r < target]
    dd = np.sqrt((downside - target).pow(2).mean()) if len(downside) else 0.0
    if dd == 0.0:
        return 0.0
    return float((r.mean() - target) / dd * np.sqrt(periods))


def calmar(equity: pd.Series, periods: int = 252) -> float:
    """年化收益 / 最大回撤绝对值。"""
    r = _returns(equity)
    if r.empty:
        return 0.0
    ann = r.mean() * periods
    peak = equity.astype(float).cummax()
    mdd = float((equity.astype(float) / peak - 1).min())
    return float(ann / abs(mdd)) if mdd != 0 else 0.0


def omega_ratio(equity: pd.Series, threshold: float = 0.0, periods: int = 252) -> float:
    """Omega 比率：下行阈值之上收益和 / 之下损失和。"""
    r = _returns(equity)
    if r.empty:
        return 0.0
    gain = r[r > threshold].sum()
    loss = -r[r < threshold].sum()
    if loss > 0:
        return float(gain / loss)
    # 无下行：有上行才是理论无穷（完美），平坦/全缺失则无信息，返回 0 避免 inf 污染下游
    return float("inf") if gain > 0 else 0.0
