"""网格交易：以中枢为基准上下均布网格，价格越低仓位越大（反向网格）。

返回整数仓位（相对中枢的网格档位数），可直接用于向量化回测的持仓输入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def grid_signal(df: pd.DataFrame, n: int = 10, range_pct: float = 0.2,
                base: float | None = None) -> pd.Series:
    c = df["close"].astype(float)
    if base is None:
        base = float(c.mean())
    lo = base * (1 - range_pct)
    hi = base * (1 + range_pct)
    levels = np.linspace(lo, hi, max(2, n))
    pos = pd.Series(0, index=df.index)
    half = n // 2
    for i in range(len(df)):
        price = float(c.iloc[i])
        idx = int(np.searchsorted(levels, price))
        lvl_pos = idx - half
        target = int(np.clip(-lvl_pos, -half, half))
        pos.iloc[i] = target
    return pos
