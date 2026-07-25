"""Vortex 指标交叉策略（迭代 113）。"""
from __future__ import annotations

import pandas as pd

from ..indicators.tech2 import vortex


def vortex_signal(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """VI+ 上穿 VI- 做多，下穿做空。"""
    v = vortex(df, period)
    sig = pd.Series(0, index=df.index)
    sig[v["vi_plus"] > v["vi_minus"]] = 1
    sig[v["vi_plus"] < v["vi_minus"]] = -1
    return sig.rename("vortex").fillna(0)
