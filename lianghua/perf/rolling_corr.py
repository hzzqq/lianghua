"""滚动相关性：两条净值曲线的滚动相关系数。"""
from __future__ import annotations

import pandas as pd


def rolling_corr_equity(eq_a: pd.Series, eq_b: pd.Series,
                          window: int = 60) -> pd.Series:
    """两条净值曲线对齐后算滚动相关。"""
    a = eq_a.astype(float)
    b = eq_b.astype(float)
    ra = a.pct_change().dropna()
    rb = b.reindex(a.index).pct_change().dropna()
    df = pd.concat([ra, rb], axis=1).dropna()
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1]).rename("roll_corr")
