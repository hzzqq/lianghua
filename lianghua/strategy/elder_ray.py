"""Elder Ray 多空力量策略（迭代 116）。"""
from __future__ import annotations

import pandas as pd


def elder_ray_signal(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """牛力(高-EMA)>0 且熊力(低-EMA)上升时做多；反之做空。"""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    ema = c.ewm(span=period, adjust=False).mean()
    bull = h - ema
    bear = l - ema
    sig = pd.Series(0, index=df.index)
    sig[(bull > 0) & (bear > bear.shift(1))] = 1
    sig[(bear < 0) & (bull < bull.shift(1))] = -1
    return sig.rename("elder_ray").fillna(0)
