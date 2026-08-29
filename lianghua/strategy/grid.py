"""网格交易：以中枢为基准上下均布网格，价格越低仓位越大（反向网格）。

返回整数仓位（相对中枢的网格档位数），可直接用于向量化回测的持仓输入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def grid_signal(df: pd.DataFrame, n: int = 10, range_pct: float = 0.2,
                base: float | None = None) -> pd.Series:
    """反向网格：相对中枢越低仓位越大，返回整数档位（可直接作为持仓）。

    base（中枢）的取值是前视偏差的关键：
    - 给定 base：固定中枢，无前视（推荐，实盘中网格中枢本就是先验参数）。
    - 缺省：改用**截至当前的历史均值**（expanding），而非全样本均值。
      原实现用 ``c.mean()`` 需要未来价格才能算出中枢，回测会系统性虚高。
    """
    c = df["close"].astype(float)
    half = n // 2
    pos = pd.Series(0, index=df.index, dtype=int)
    centers = (
        pd.Series(float(base), index=df.index)
        if base is not None
        else c.expanding().mean()
    )
    for i in range(len(df)):
        b = float(centers.iloc[i])
        price = float(c.iloc[i])
        if not np.isfinite(b) or b <= 0 or not np.isfinite(price):
            continue
        levels = np.linspace(b * (1 - range_pct), b * (1 + range_pct), max(2, n))
        idx = int(np.searchsorted(levels, price))
        pos.iloc[i] = int(np.clip(-(idx - half), -half, half))
    return pos
