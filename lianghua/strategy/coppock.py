"""科普克曲线（Coppock Curve）。"""
from __future__ import annotations

import pandas as pd


def coppock(close: pd.Series, w1: int = 11, w2: int = 14,
              roc_months: int = 10) -> pd.Series:
    """月度 ROC 加权和的指数平滑。"""
    c = close.astype(float)
    if not isinstance(c.index, pd.DatetimeIndex):
        # 引擎传入的 df 多为默认 RangeIndex；合成等距日频索引以支撑 resample
        c = c.set_axis(pd.date_range("2000-01-01", periods=len(c), freq="D"))
    monthly = c.resample("ME").last()
    roc = (monthly / monthly.shift(roc_months) - 1) * 100
    wma = (roc * w1 + roc.shift(1) * w2)
    return wma.ewm(span=10, adjust=False).mean().rename("coppock")


def coppock_signal(close: pd.Series) -> pd.Series:
    """曲线由负转正买入（持有至转负）。"""
    cp = coppock(close)
    sig = pd.Series(0, index=cp.index)
    sig[cp > 0] = 1
    return sig.reindex(close.index).ffill().fillna(0).rename("coppock_sig")
