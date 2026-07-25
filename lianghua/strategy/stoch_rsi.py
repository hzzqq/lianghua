"""Stochastic RSI 超买超卖策略（迭代 120）。"""
from __future__ import annotations

import pandas as pd

from ..indicators.tech2 import stoch_rsi


def stoch_rsi_signal(close: pd.Series, period: int = 14, k: int = 3, d: int = 3,
                     low: float = 0.2, high: float = 0.8) -> pd.Series:
    """K 线自低位上穿 D 做多，自高位下穿 D 做空。"""
    sr = stoch_rsi(close, period, k, d)
    kk, dd = sr["k"], sr["d"]
    sig = pd.Series(0, index=close.index)
    sig[(kk < low) & (kk > dd)] = 1
    sig[(kk > high) & (kk < dd)] = -1
    return sig.rename("stoch_rsi").fillna(0)
