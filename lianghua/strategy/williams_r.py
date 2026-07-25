"""Williams %R 超买超卖反转策略（迭代 115）。"""
from __future__ import annotations

import pandas as pd

from ..indicators.tech2 import williams_r


def williams_r_signal(df: pd.DataFrame, period: int = 14,
                      low: float = -80.0, high: float = -20.0) -> pd.Series:
    """%R 低于 -80 超卖做多，高于 -20 超买做空。"""
    wr = williams_r(df, period)
    sig = pd.Series(0, index=df.index)
    sig[wr < low] = 1
    sig[wr > high] = -1
    return sig.rename("williams_r").fillna(0)
