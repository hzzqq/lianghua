"""波动率突破：收益异常放大时顺势。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_breakout_signal(close: pd.Series, window: int = 20,
                          mult: float = 2.0) -> pd.Series:
    """单日收益超过 mult×滚动波动则发方向信号（沿用至上次信号）。"""
    r = close.astype(float).pct_change()
    sigma = r.rolling(window).std()
    z = r / (sigma + 1e-12)
    raw = pd.Series(0, index=r.index)
    raw[z > mult] = 1
    raw[z < -mult] = -1
    # 信号保持到下一次反向突破
    sig = pd.Series(0, index=r.index)
    pos = 0
    for i in range(len(raw)):
        if raw.iloc[i] != 0:
            pos = raw.iloc[i]
        sig.iloc[i] = pos
    return sig.rename("vol_breakout")
