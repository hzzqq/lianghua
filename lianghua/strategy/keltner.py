"""Keltner 通道信号。"""
from __future__ import annotations

import pandas as pd


def keltner_signal(close: pd.Series, high: pd.Series, low: pd.Series,
                     window: int = 20, mult: float = 2.0) -> pd.Series:
    """中轨=EMA，上下轨=中轨±mult*ATR。收盘站上轨做多、跌破下轨做空。"""
    c = close.astype(float)
    mid = c.ewm(span=window, adjust=False).mean()
    tr = (high.astype(float) - low.astype(float))
    atr = tr.rolling(window).mean()
    upper = mid + mult * atr
    lower = mid - mult * atr
    sig = pd.Series(0, index=c.index)
    sig[c > upper] = 1
    sig[c < lower] = -1
    return sig.rename("keltner")
