"""因子预处理：缩尾 / 标准化 / 行业中性化。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """按分位缩尾，抑制极端值。"""
    s = series.astype(float)
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def zscore(series: pd.Series) -> pd.Series:
    """标准化（去均值 / 单位标准差）。"""
    s = series.astype(float)
    return ((s - s.mean()) / (s.std() + 1e-12)).rename("z")


def neutralize(factor: pd.Series, industry: pd.Series) -> pd.Series:
    """行业中性化：对行业哑变量回归取残差。"""
    f = factor.astype(float)
    ind_series = industry.astype(str).rename("ind")
    dummies = pd.get_dummies(pd.Categorical(ind_series))
    X = pd.concat([pd.Series(1.0, index=f.index, name="const"), dummies], axis=1).astype(float)
    y = f.values
    Xv = X.values
    try:
        beta = np.linalg.lstsq(Xv, y, rcond=None)[0]
        resid = y - Xv @ beta
    except Exception:
        resid = y - y.mean()
    return pd.Series(resid, index=f.index, name="neutral")
