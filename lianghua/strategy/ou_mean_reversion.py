"""OU 过程均值回归信号。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ou_signal(price: pd.Series, window: int = 60, entry: float = 2.0,
               exit: float = 0.5) -> pd.Series:
    """价格相对滚动均值的 z-score；超卖做多、超买卖出。"""
    p = price.astype(float)
    z = (p - p.rolling(window).mean()) / (p.rolling(window).std() + 1e-12)
    sig = pd.Series(0, index=p.index)
    pos = 0
    for i in range(len(z)):
        if pd.isna(z.iloc[i]):
            sig.iloc[i] = pos
            continue
        if pos == 0 and z.iloc[i] < -entry:
            pos = 1
        elif pos == 0 and z.iloc[i] > entry:
            pos = -1
        elif pos != 0 and abs(z.iloc[i]) < exit:
            pos = 0
        sig.iloc[i] = pos
    return sig.rename("ou")
