"""Pivot Points 轴心点策略（迭代 141）。

以经典轴心点(Pivot)为多空分界：收盘价站上 Pivot 看多(+1)，跌破看空(-1)，
否则观望(0)。输入 OHLCV DataFrame，输出等长 {-1,0,1} 信号序列。
纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pivot_points_signal(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    pivot = (high + low + close) / 3.0
    sig = np.where(
        close > pivot, 1,
        np.where(close < pivot, -1, 0),
    )
    return pd.Series(sig, index=df.index, name="pivot_points")
