"""基准对比增强：超额收益 / 跟踪误差 / 信息比率。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def excess_return(equity: pd.Series, benchmark: pd.Series, annual: int = 252) -> float:
    if annual <= 0:
        return 0.0
    r = equity.pct_change().dropna()
    b = benchmark.pct_change().dropna()
    common = r.index.intersection(b.index)
    if common.empty:
        return 0.0
    m = float((r[common] - b[common]).mean() * annual)
    return 0.0 if pd.isna(m) else m


def tracking_error(equity: pd.Series, benchmark: pd.Series, annual: int = 252) -> float:
    if annual <= 0:
        return 0.0
    r = equity.pct_change().dropna()
    b = benchmark.pct_change().dropna()
    common = r.index.intersection(b.index)
    if common.empty:
        return 0.0
    s = float((r[common] - b[common]).std() * np.sqrt(annual))
    return 0.0 if pd.isna(s) else s


def information_ratio(equity: pd.Series, benchmark: pd.Series) -> float:
    te = tracking_error(equity, benchmark)
    return excess_return(equity, benchmark) / te if te > 0 else 0.0
