"""Tearsheet 风格绩效摘要表。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lianghua.perf.annualize import annual_return, annual_vol
from lianghua.perf.sortino import calmar, sortino
from lianghua.risk.drawdown import max_drawdown_info


def _finite(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")


def _safe(fn, default=float("nan")):
    """安全执行指标计算：下游函数对 NaN/非正值会抛错或产出 inf，统一降级为默认值。"""
    try:
        v = fn()
    except Exception:
        return default
    return v if _finite(v) else default


def tearsheet(equity: pd.Series, periods: int = 252) -> pd.DataFrame:
    """返回单列多指标的 Tearsheet。"""
    eq = equity.astype(float)
    if len(eq) == 0:
        return pd.DataFrame({"指标": [], "数值": []})
    first = eq.iloc[0]
    last = eq.iloc[-1]
    cum = (last / first - 1) * 100 if (_finite(first) and first != 0) else float("nan")
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    std = r.std() if len(r) else float("nan")
    sharpe = (r.mean() / std * (periods ** 0.5)) if (_finite(std) and std != 0) else 0.0
    rows = {
        "累计收益%": float(cum) if _finite(cum) else float("nan"),
        "年化收益%": _safe(lambda: annual_return(eq, periods) * 100)
        if (_finite(first) and first != 0)
        else float("nan"),
        "年化波动%": _safe(lambda: annual_vol(r, periods) * 100),
        "夏普": sharpe,
        "索提诺": _safe(lambda: sortino(eq)),
        "卡玛": _safe(lambda: calmar(eq, periods)),
        "最大回撤%": _safe(lambda: max_drawdown_info(eq)["max_drawdown"] * 100),
        "回撤天数": _safe(lambda: max_drawdown_info(eq)["duration_days"], default=0),
    }
    return pd.DataFrame({"指标": list(rows.keys()), "数值": list(rows.values())})
