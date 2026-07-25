"""自适应：波动率区制切换风险敞口。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adaptive_vol_switch(ret: pd.Series, fast: int = 20, slow: int = 60,
                          threshold: float = 1.2) -> pd.Series:
    """快/慢波动比超阈值视为高波动区 → 风险回避（0），否则风险开启（1）。"""
    r = ret.astype(float)
    vf = r.rolling(fast).std()
    vs = r.rolling(slow).std()
    ratio = vf / (vs + 1e-12)
    return pd.Series(np.where(ratio > threshold, 0, 1), index=r.index, name="risk_on")


def adaptive_position(ret: pd.Series, target_vol: float = 0.10,
                         periods: int = 252) -> pd.Series:
    """按目标波动动态缩放仓位（波动越高仓位越低）。"""
    r = ret.astype(float)
    vol = r.rolling(20).std() * np.sqrt(periods)
    w = target_vol / (vol + 1e-9)
    return w.clip(0, 1).rename("adaptive_w")
