"""Supertrend 趋势跟踪策略（迭代 111）。"""
from __future__ import annotations

import pandas as pd

from ..indicators.tech2 import supertrend


def supertrend_signal(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """方向翻多做多(+1)、翻空做空(-1)。"""
    st = supertrend(df, period, mult)
    sig = st["dir"].astype(int)
    return pd.Series(sig.values, index=df.index).rename("supertrend").fillna(0)
