"""Klinger Volume Oscillator 量价策略（迭代 145）。

以「高低收合计」的趋势方向与成交量构造 Force，取快/慢 EMA 之差(KO)与其信号线，
KO 上穿信号线看多(+1)、下穿看空(-1)。输入 OHLCV DataFrame。
纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def klinger_signal(df: pd.DataFrame, fast: int = 34, slow: int = 55,
                   signal: int = 13) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    prev_close = close.shift()
    hlc = high + low + close
    hlc_prev = hlc.shift()
    trend = np.sign((hlc - hlc_prev).fillna(0.0).values)
    force = (volume.values * trend)

    f = pd.Series(force, index=df.index)
    ko = f.ewm(span=fast, adjust=False).mean() - f.ewm(span=slow, adjust=False).mean()
    sig_line = ko.ewm(span=signal, adjust=False).mean()
    sig = np.where(
        ko > sig_line, 1,
        np.where(ko < sig_line, -1, 0),
    )
    return pd.Series(sig, index=df.index, name="klinger")
