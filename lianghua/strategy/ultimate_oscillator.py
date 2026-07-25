"""Ultimate Oscillator 终极波动策略（迭代 135）。
三周期加权波动低于 30 超卖做多、高于 70 超买做空。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ultimate_oscillator(df: pd.DataFrame) -> pd.Series:
    close, low, high = df["close"], df["low"], df["high"]
    prev_close = close.shift()
    bp = close - np.minimum(low, prev_close)
    tr = pd.Series(np.maximum.reduce([
        (high - low).abs().values,
        (high - prev_close).abs().values,
        (low - prev_close).abs().values,
    ]), index=df.index)

    def avg(p: int) -> pd.Series:
        return bp.rolling(p).sum() / tr.rolling(p).sum()

    a1, a2, a3 = avg(7), avg(14), avg(28)
    uo = 100 * (4 * a1 + 2 * a2 + a3) / 7
    return uo


def ultimate_oscillator_signal(df: pd.DataFrame, low: float = 30.0,
                               high: float = 70.0) -> pd.Series:
    """UO 超卖带做多、超买带做空。"""
    uo = _ultimate_oscillator(df)
    sig = pd.Series(0, index=df.index)
    sig[uo < low] = 1
    sig[uo > high] = -1
    return sig.rename("ultimate_oscillator").fillna(0)
