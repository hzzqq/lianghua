"""双重动量：相对动量选资产 + 绝对动量决定是否持有。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dual_momentum(asset_ret: pd.DataFrame, lookback: int = 120,
                   cash_col: str | None = None) -> pd.Series:
    """返回持仓资产名序列（相对动量最强者；若其累计收益<=0 则切现金）。"""
    r = asset_ret.astype(float)
    cols = [c for c in r.columns if c != cash_col]
    if not cols:
        return pd.Series(dtype=object)
    cum = (1 + r[cols]).rolling(lookback).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    best = cum.fillna(-np.inf).idxmax(axis=1)
    pos = best.where(cum.max(axis=1) > 0, other=(cash_col or "CASH"))
    return pos.rename("position")
