"""ADX 趋势强度策略（迭代 132）。
+DI 上穿 -DI 且 ADX>阈值（趋势明确）做多；反之做空。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _adx(df: pd.DataFrame, period: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev_close = close.shift()
    tr = pd.Series(np.maximum.reduce([
        (high - low).abs().values,
        (high - prev_close).abs().values,
        (low - prev_close).abs().values,
    ]), index=df.index)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return plus_di, minus_di, adx


def adx_trend_signal(df: pd.DataFrame, period: int = 14,
                     th: float = 25.0) -> pd.Series:
    """趋势明确时顺势：+DI>-DI 多，否则空。"""
    plus_di, minus_di, adx = _adx(df, period)
    sig = pd.Series(0, index=df.index)
    mask = adx > th
    sig[mask & (plus_di > minus_di)] = 1
    sig[mask & (minus_di > plus_di)] = -1
    return sig.rename("adx_trend").fillna(0)
