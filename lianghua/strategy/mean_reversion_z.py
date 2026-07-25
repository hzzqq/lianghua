"""Z-score 均值回归策略：价格偏离均线超过 entry 个标准差反向，回到 exit 内平仓。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_signal(df: pd.DataFrame, period: int = 20, entry: float = 2.0,
                  exit_: float = 0.5) -> pd.Series:
    c = df["close"].astype(float)
    m = c.rolling(period).mean()
    sd = c.rolling(period).std()
    z = (c - m) / (sd + 1e-12)
    sig = pd.Series(0, index=df.index)
    sig[z < -entry] = 1
    sig[z > entry] = -1
    sig[(z > -exit_) & (z < exit_)] = 0
    return sig
