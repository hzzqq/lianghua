"""Aroon 趋势策略（迭代 112）。"""
from __future__ import annotations

import pandas as pd

from ..indicators.tech2 import aroon


def aroon_signal(df: pd.DataFrame, period: int = 25, th: float = 70.0) -> pd.Series:
    """Aroon-Up 强于 Aroon-Down 且 >th 做多，反之做空。"""
    a = aroon(df, period)
    sig = pd.Series(0, index=df.index)
    sig[(a["up"] > th) & (a["up"] > a["down"])] = 1
    sig[(a["down"] > th) & (a["down"] > a["up"])] = -1
    return sig.rename("aroon").fillna(0)
