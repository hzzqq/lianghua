"""唐奇安通道突破。"""
from __future__ import annotations

import pandas as pd


def donchian_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """收盘突破上轨做多、跌破下轨做空。"""
    c = close.astype(float)
    upper = c.rolling(window).max().shift(1)
    lower = c.rolling(window).min().shift(1)
    sig = pd.Series(0, index=c.index)
    sig[c > upper] = 1
    sig[c < lower] = -1
    return sig.rename("donchian")
