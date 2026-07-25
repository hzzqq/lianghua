"""动量轮动：每期根据过去 lookback 收益，等权配置表现最好的 top 个资产。

返回权重 DataFrame（index=日期, columns=资产），可直接用于篮子/组合回测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rotation_weights(returns: pd.DataFrame, top: int = 1, lookback: int = 60) -> pd.DataFrame:
    cols = list(returns.columns)
    w = pd.DataFrame(index=returns.index, columns=cols, data=0.0)
    n = len(returns)
    for i in range(lookback, n):
        mom = returns.iloc[i - lookback:i].sum()
        best = mom.nlargest(top).index.tolist()
        for b in best:
            w.iloc[i, cols.index(b)] = 1.0 / len(best)
    return w
