"""布林带均值回归策略。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bollinger_signal(df: pd.DataFrame, period: int = 20, nb: float = 2.0) -> pd.Series:
    c = df["close"].astype(float)
    m = c.rolling(period).mean()
    sd = c.rolling(period).std()
    up = m + nb * sd
    dn = m - nb * sd
    sig = pd.Series(0, index=df.index)
    sig[c < dn] = 1      # 触下轨买入
    sig[c > up] = -1     # 触上轨卖出
    return sig
