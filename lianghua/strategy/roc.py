"""ROC 变动率动量策略（迭代 134）。
12 日变动率为正做多、为负做空。
"""
from __future__ import annotations

import pandas as pd


def roc_signal(close: pd.Series, period: int = 12,
               th: float = 0.0) -> pd.Series:
    """ROC 动量方向：正动量做多，负动量做空。"""
    roc = close.pct_change(period)
    sig = pd.Series(0, index=close.index)
    sig[roc > th] = 1
    sig[roc < -th] = -1
    return sig.rename("roc").fillna(0)
