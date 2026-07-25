"""海龟交易法则（Turtle）：唐奇安通道突破入场 + ATR 通道出场。

返回带符号仓位序列（-1/0/1），可直接喂给向量化回测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def donchian(df: pd.DataFrame, entry: int = 20, exit_: int = 10):
    c = df["close"].astype(float)
    hi = c.rolling(entry).max()
    lo = c.rolling(entry).min()
    exh = c.rolling(exit_).max()
    exl = c.rolling(exit_).min()
    return hi, lo, exh, exl


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev).abs(), (l - prev).abs()))
    return tr.rolling(period).mean()


def turtle_signals(df: pd.DataFrame, entry: int = 20, exit_: int = 10,
                   atr_period: int = 14) -> pd.Series:
    """海龟信号：上破 entry 日高点做多，下破 exit 日低点平多；反之做空。"""
    c = df["close"].astype(float)
    hi, lo, exh, exl = donchian(df, entry, exit_)
    pos = pd.Series(0, index=df.index)
    cur = 0
    for i in range(1, len(df)):
        if cur == 0:
            if c.iloc[i] >= hi.iloc[i - 1]:
                cur = 1
            elif c.iloc[i] <= lo.iloc[i - 1]:
                cur = -1
        elif cur > 0:
            if c.iloc[i] <= exl.iloc[i - 1]:
                cur = 0
        else:
            if c.iloc[i] >= exh.iloc[i - 1]:
                cur = 0
        pos.iloc[i] = cur
    return pos
