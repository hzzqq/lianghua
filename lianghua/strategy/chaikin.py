"""Chaikin Money Flow 资金流策略（迭代 119）。"""
from __future__ import annotations

import pandas as pd

from ..indicators.tech2 import cmf


def chaikin_signal(df: pd.DataFrame, period: int = 20, th: float = 0.05) -> pd.Series:
    """CMF > th 资金流入做多，< -th 流出做空。"""
    f = cmf(df, period)
    sig = pd.Series(0, index=df.index)
    sig[f > th] = 1
    sig[f < -th] = -1
    return sig.rename("chaikin").fillna(0)
