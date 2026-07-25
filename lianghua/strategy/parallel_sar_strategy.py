"""Parabolic SAR 趋势策略（迭代 131）。
基于 Wilder 抛物线 SAR，close 上穿 SAR 做多、下穿做空。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _psar(high: pd.Series, low: pd.Series, close: pd.Series,
          step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    n = len(high)
    if n < 2:
        return pd.Series(0, index=close.index)
    sar = np.zeros(n)
    trend = np.ones(n, dtype=int)  # 1 多头, -1 空头
    ep = np.zeros(n)
    af = np.zeros(n)
    trend[0], ep[0], sar[0], af[0] = 1, high.iloc[0], low.iloc[0], step
    for i in range(1, n):
        if trend[i - 1] == 1:
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            if low.iloc[i] < sar[i]:
                trend[i], sar[i], ep[i], af[i] = -1, ep[i - 1], low.iloc[i], step
            else:
                trend[i] = 1
                if high.iloc[i] > ep[i - 1]:
                    ep[i], af[i] = high.iloc[i], min(af[i - 1] + step, max_step)
                else:
                    ep[i], af[i] = ep[i - 1], af[i - 1]
                sar[i] = min(sar[i], low.iloc[i - 1])
        else:
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])
            if high.iloc[i] > sar[i]:
                trend[i], sar[i], ep[i], af[i] = 1, ep[i - 1], high.iloc[i], step
            else:
                trend[i] = -1
                if low.iloc[i] < ep[i - 1]:
                    ep[i], af[i] = low.iloc[i], min(af[i - 1] + step, max_step)
                else:
                    ep[i], af[i] = ep[i - 1], af[i - 1]
                sar[i] = max(sar[i], high.iloc[i - 1])
    return pd.Series(trend, index=close.index)


def parabolic_sar_signal(df: pd.DataFrame, step: float = 0.02,
                         max_step: float = 0.2) -> pd.Series:
    """SAR 在价格下方=多头(1)，上方=空头(-1)。"""
    sig = _psar(df["high"], df["low"], df["close"], step, max_step)
    return sig.rename("parabolic_sar").fillna(0)
