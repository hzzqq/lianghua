"""月度收益分析：月度收益序列 + 年×月热力表。"""
from __future__ import annotations

import pandas as pd


def monthly_returns(equity: pd.Series) -> pd.Series:
    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    return eq.resample("ME").last().pct_change().dropna()


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    m = eq.resample("ME").last().pct_change().dropna()
    tbl = m.to_frame("月收益")
    tbl["年"] = tbl.index.year
    tbl["月"] = tbl.index.month
    return tbl.pivot(index="年", columns="月", values="月收益").fillna(0.0)
