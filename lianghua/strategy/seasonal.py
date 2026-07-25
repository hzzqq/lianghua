"""季节性（月份效应）信号。"""
from __future__ import annotations

import pandas as pd


def month_signal(close: pd.Series, strong_months=(11, 12, 1, 2)) -> pd.Series:
    """在强势月份持有（1），其余空仓（0）。"""
    idx = pd.to_datetime(close.index)
    return pd.Series([1 if m in strong_months else 0 for m in idx.month],
                      index=close.index, name="seasonal")


def month_momentum(close: pd.Series) -> pd.Series:
    """统计各月平均收益，持有历史正收益月。"""
    c = close.astype(float)
    c.index = pd.to_datetime(c.index)
    mret = c.resample("ME").last().pct_change()
    avg = mret.groupby(mret.index.month).mean()
    keep = set(avg[avg > 0].index)
    idx = pd.to_datetime(close.index)
    return pd.Series([1 if m in keep else 0 for m in idx.month],
                      index=close.index, name="month_mom")
