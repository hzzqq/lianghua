"""RSI 反转策略：超卖买入、超买卖出。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.astype(float).diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    rs = up.rolling(period).mean() / (dn.rolling(period).mean() + 1e-12)
    return 100 - 100 / (1 + rs)


def rsi_signal(df: pd.DataFrame, period: int = 14, buy: float = 30.0,
               sell: float = 70.0) -> pd.Series:
    r = rsi(df["close"], period)
    return pd.Series(np.where(r < buy, 1, np.where(r > sell, -1, 0)), index=df.index)
