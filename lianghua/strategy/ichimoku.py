"""一目均衡表（Ichimoku）云图策略（迭代 117）。"""
from __future__ import annotations

import pandas as pd


def ichimoku_signal(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
                    senkou: int = 52) -> pd.Series:
    """价格在云上方且转换线上穿基准线做多；云下方且下穿做空。"""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    conv = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2
    base = (h.rolling(kijun).max() + l.rolling(kijun).min()) / 2
    span_a = ((conv + base) / 2).shift(kijun)
    span_b = ((h.rolling(senkou).max() + l.rolling(senkou).min()) / 2).shift(kijun)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    sig = pd.Series(0, index=df.index)
    sig[(c > cloud_top) & (conv > base)] = 1
    sig[(c < cloud_bot) & (conv < base)] = -1
    return sig.rename("ichimoku").fillna(0)
