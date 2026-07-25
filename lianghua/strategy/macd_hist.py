"""MACD 柱状图零轴穿越策略（迭代 118）。"""
from __future__ import annotations

import pandas as pd

from ..indicators.tech import macd


def macd_hist_signal(close: pd.Series, fast: int = 12, slow: int = 26,
                     signal: int = 9) -> pd.Series:
    """MACD 柱(hist)由负转正做多，由正转负做空。"""
    m = macd(close, fast, slow, signal)
    hist = m["hist"]
    sig = pd.Series(0, index=close.index)
    sig[hist > 0] = 1
    sig[hist < 0] = -1
    return sig.rename("macd_hist").fillna(0)
