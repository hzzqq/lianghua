"""CPPI 固定比例投资组合保险：底仓保护 + 风险乘数。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cppi(asset_returns: pd.Series, floor_pct: float = 0.8,
           multiplier: float = 3.0, init_cash: float = 1_000_000.0,
           risk_free: float = 0.0, periods: int = 252) -> pd.DataFrame:
    """返回含 equity / cushion / risky_weight 的 DataFrame。"""
    r = asset_returns.astype(float).reset_index(drop=True)
    n = len(r)
    floor_val = floor_pct * init_cash
    value = init_cash
    rows = []
    for t in range(n):
        floor_val *= (1 + risk_free / periods)
        cushion = value - floor_val
        risky_w = max(0.0, min(1.0, multiplier * cushion / value)) if value > 0 else 0.0
        rows.append({"value": value, "cushion": cushion, "risky_weight": risky_w})
        value = value * (risky_w * (1 + r.iloc[t]) + (1 - risky_w))
    df = pd.DataFrame(rows)
    df["equity"] = df["value"]
    return df[["equity", "cushion", "risky_weight"]]
