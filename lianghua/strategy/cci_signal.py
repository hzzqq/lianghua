"""CCI 均值回复策略（迭代 133）。
CCI 跌破 -100 视为超卖做多，突破 +100 视为超买做空。
"""
from __future__ import annotations

import pandas as pd

from ..indicators.tech import cci


def cci_signal(df: pd.DataFrame, period: int = 20,
               low: float = -100.0, high: float = 100.0) -> pd.Series:
    """CCI 进入超卖/超买带触发反向信号。"""
    c = cci(df, period)
    sig = pd.Series(0, index=df.index)
    sig[c < low] = 1
    sig[c > high] = -1
    return sig.rename("cci_signal").fillna(0)
