"""DeMark 式比较策略（迭代 144）。

参考 Tom DeMark 的「比较收盘价与 N 期前收盘价」思路：
close > close[N] 视为买入动能(+1)，反之下跌(-1)。缺省 N=4。
输入 OHLCV DataFrame，输出等长 {-1,0,1} 信号。
纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def demark_signal(df: pd.DataFrame, lookback: int = 4) -> pd.Series:
    close = df["close"].astype(float)
    prev = close.shift(lookback)
    sig = np.where(
        close > prev, 1,
        np.where(close < prev, -1, 0),
    )
    return pd.Series(sig, index=df.index, name="demark")
