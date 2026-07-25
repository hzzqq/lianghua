"""TRIX 动量交叉策略（迭代 114）。"""
from __future__ import annotations

import pandas as pd

from ..indicators.tech2 import trix


def trix_signal(close: pd.Series, period: int = 15, signal: int = 9) -> pd.Series:
    """TRIX 上穿信号线做多，下穿做空。"""
    t = trix(close, period, signal)
    sig = pd.Series(0, index=close.index)
    sig[t["trix"] > t["signal"]] = 1
    sig[t["trix"] < t["signal"]] = -1
    return sig.rename("trix").fillna(0)
