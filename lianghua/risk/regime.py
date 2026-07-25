"""波动率区制识别：高低波动切换。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_regime(returns: pd.Series, slow: int = 60,
                    mult: float = 1.5) -> pd.Series:
    """滚动波动超过慢线 mult 倍 → 高波动区(1)，否则低波动(0)。"""
    r = returns.astype(float)
    vf = r.rolling(20).std()
    vs = r.rolling(slow).std()
    th = vs * mult
    return pd.Series(np.where(vf > th, 1, 0), index=r.index, name="regime")


def smooth_regime(regime: pd.Series, min_hold: int = 5) -> pd.Series:
    """抑制区制频繁抖动：连续持有至少 min_hold 根 bar 才允许切换。

    高频假突破会导致高/低波动反复横跳、触发重复调仓。以"最短持有"为闸门，
    仅在当前状态已持续 >= min_hold 后才接受切换。新增能力。
    """
    reg = regime.astype(int).reset_index(drop=True)
    out = reg.copy()
    hold = 0
    for i in range(1, len(reg)):
        if reg[i] == reg[i - 1]:
            hold += 1
        else:
            if hold < min_hold:
                out[i] = reg[i - 1]   # 持有不足，维持旧状态
            else:
                hold = 0              # 允许切换，重置计数
    out.index = regime.index
    return out.rename("regime_smooth")


def regime_stats(returns: pd.Series, regime: pd.Series) -> pd.DataFrame:
    """分 regime 统计收益均值/波动。"""
    r = returns.astype(float)
    reg = regime.reindex(r.index).fillna(0).astype(int)
    df = pd.concat([r, reg], axis=1)
    df.columns = ["ret", "regime"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.groupby("regime")["ret"].agg(["mean", "std", "count"])
