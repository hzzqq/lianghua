"""RSI 背离信号（动量衰竭）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_divergence(close: pd.Series, rsi: pd.Series,
                      lookback: int = 30) -> pd.Series:
    """价格创新高但 RSI 未创新高 → 顶背离(-1)；反之亦然(+1)。"""
    c = close.astype(float)
    rs = rsi.astype(float)
    sig = pd.Series(0, index=c.index)
    for i in range(lookback, len(c)):
        pc = c.iloc[i - lookback:i]
        pr = rs.iloc[i - lookback:i]
        if c.iloc[i] > pc.max() and rs.iloc[i] < pr.max():
            sig.iloc[i] = -1
        elif c.iloc[i] < pc.min() and rs.iloc[i] > pr.min():
            sig.iloc[i] = 1
    return sig.rename("rsi_div")
