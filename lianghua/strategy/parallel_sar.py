"""抛物线 SAR（停损反转）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parabolic_sar(high: pd.Series, low: pd.Series,
                    af: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """返回 SAR 序列（与价格同 index）。"""
    h = high.astype(float).values
    l = low.astype(float).values
    n = len(h)
    sar = np.zeros(n)
    # 初始趋势：前两日决定
    uptrend = h[1] > h[0]
    sar[0] = l[0] if uptrend else h[0]
    ep = h[0] if uptrend else l[0]
    aaf = af
    for t in range(1, n):
        prev = sar[t - 1]
        if uptrend:
            sar[t] = prev + aaf * (ep - prev)
            if l[t] < sar[t]:
                uptrend = False
                sar[t] = ep
                ep = l[t]
                aaf = af
            else:
                if h[t] > ep:
                    ep = h[t]
                    aaf = min(aaf + af, af_max)
        else:
            sar[t] = prev + aaf * (ep - prev)
            if h[t] > sar[t]:
                uptrend = True
                sar[t] = ep
                ep = h[t]
                aaf = af
            else:
                if l[t] < ep:
                    ep = l[t]
                    aaf = min(aaf + af, af_max)
    return pd.Series(sar, index=high.index, name="sar")


def sar_signal(high: pd.Series, low: pd.Series) -> pd.Series:
    sar = parabolic_sar(high, low)
    close = (high.astype(float) + low.astype(float)) / 2
    return pd.Series(np.where(close > sar, 1, -1), index=sar.index, name="sar_sig")
