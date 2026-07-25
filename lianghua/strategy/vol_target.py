"""波动率目标配置：将组合年化波动约束到 target，按近期波动反向缩放。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_weights(returns: pd.DataFrame, target: float = 0.15,
                       window: int = 20, annual: int = 252) -> pd.DataFrame:
    cols = list(returns.columns)
    w = pd.DataFrame(index=returns.index, columns=cols, data=0.0)
    vol = returns.rolling(window).std() * np.sqrt(annual)
    n = len(returns)
    for i in range(window, n):
        v = vol.iloc[i].replace(0, np.nan)
        scale = (target / v).clip(upper=3.0).fillna(0.0)
        tot = scale.sum()
        if tot > 0:
            w.iloc[i] = scale / tot
    return w
