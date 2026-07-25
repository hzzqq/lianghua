"""事件研究：异常收益（AR）与累计异常收益（CAR）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def event_study(returns: pd.Series, events: list, window: int = 10) -> pd.DataFrame:
    """events: 事件日列表(datetime 兼容)；返回每事件的 AR 与 CAR。"""
    r = returns.astype(float)
    r.index = pd.to_datetime(r.index)
    rows = []
    for e in events:
        e = pd.Timestamp(e)
        lo = e - pd.Timedelta(days=window * 2)
        pre = r[lo:e]
        mu = pre.mean() if len(pre) else 0.0
        seg = r[e: e + pd.Timedelta(days=window * 2)]
        ar = (seg - mu).dropna()
        car = ar.cumsum()
        rows.append({
            "event": e,
            "AR_mean": float(ar.mean()),
            "CAR_end": float(car.iloc[-1]) if len(car) else 0.0,
            "n_obs": int(len(ar)),
        })
    return pd.DataFrame(rows)
