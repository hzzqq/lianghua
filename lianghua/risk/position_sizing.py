"""仓位管理：凯利 / 固定分数。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(win_rate: float, win_odds: float) -> float:
    """Kelly = p - (1-p)/b。win_odds=b=盈利/亏损比。"""
    p = win_rate
    b = win_odds
    if b <= 0:
        return 0.0
    f = p - (1 - p) / b
    return float(max(0.0, min(1.0, f)))


def fixed_fractional(equity: float, risk_pct: float, stop_dist: float) -> float:
    """每笔风险 = equity*risk_pct，仓位 = 风险/止损距离。"""
    if stop_dist <= 0:
        return 0.0
    return float(equity * risk_pct / stop_dist)


def vol_target_size(equity: float, price: float, daily_vol: float,
                     target_vol: float = 0.01, periods: int = 252) -> float:
    """使单标的日波动≈target 的股数。"""
    if price <= 0 or daily_vol <= 0:
        return 0.0
    port_vol = target_vol * equity
    unit_vol = price * daily_vol
    return float(port_vol / unit_vol)
