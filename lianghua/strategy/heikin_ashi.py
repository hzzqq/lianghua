"""Heikin-Ashi 云图趋势策略（迭代 142）。

把 K 线转换为 Heikin-Ashi，以 HA_close 与 HA_open 的关系判断趋势：
HA_close > HA_open 视为上升(多头 +1)，反之下跌(-1)。输入 OHLCV DataFrame。
纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def heikin_ashi_signal(df: pd.DataFrame) -> pd.Series:
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    ha_close = (open_ + high + low + close) / 4.0
    ha_open = pd.Series(index=df.index, dtype=float)
    prev = (open_.iloc[0] + close.iloc[0]) / 2.0
    for i in range(len(df)):
        if i == 0:
            o = prev
        else:
            o = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
        ha_open.iloc[i] = o

    trend = ha_close - ha_open
    sig = np.where(trend > 0, 1, np.where(trend < 0, -1, 0))
    return pd.Series(sig, index=df.index, name="heikin_ashi")
