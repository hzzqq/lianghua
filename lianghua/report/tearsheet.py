"""Tearsheet 风格绩效摘要表。"""
from __future__ import annotations

import pandas as pd

from lianghua.perf.annualize import annual_return, annual_vol
from lianghua.perf.sortino import calmar, sortino
from lianghua.risk.drawdown import max_drawdown_info


def tearsheet(equity: pd.Series, periods: int = 252) -> pd.DataFrame:
    """返回单列多指标的 Tearsheet。"""
    eq = equity.astype(float)
    r = eq.pct_change().dropna()
    info = max_drawdown_info(eq)
    rows = {
        "累计收益%": float((eq.iloc[-1] / eq.iloc[0] - 1) * 100),
        "年化收益%": annual_return(eq, periods) * 100,
        "年化波动%": annual_vol(r, periods) * 100,
        "夏普": (r.mean() / r.std() * (periods ** 0.5)) if r.std() else 0.0,
        "索提诺": sortino(eq),
        "卡玛": calmar(eq, periods),
        "最大回撤%": info["max_drawdown"] * 100,
        "回撤天数": info["duration_days"],
    }
    return pd.DataFrame({"指标": list(rows.keys()), "数值": list(rows.values())})
