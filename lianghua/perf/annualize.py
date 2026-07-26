"""年度化工具：收益 / 波动 / 比率换算。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annual_return(equity: pd.Series, periods: int = 252) -> float:
    eq = equity.astype(float)
    if len(eq) < 2 or periods <= 0:
        return 0.0
    total = eq.iloc[-1] / eq.iloc[0]
    # 净值跌破/归零时负底数做分数幂会抛 ValueError，安全降级
    if total <= 0:
        return 0.0
    years = len(eq) / periods
    return float(total ** (1 / years) - 1) if years > 0 else 0.0


def annual_vol(returns: pd.Series, periods: int = 252) -> float:
    r = returns.astype(float)
    return float(r.std() * np.sqrt(periods)) if len(r) else 0.0


def annualize_ratio(ratio: float, from_periods: int, to_periods: int = 252) -> float:
    """把 from_periods 频率下的比率换算到 to_periods。"""
    if from_periods <= 0 or to_periods <= 0:
        return 0.0
    return float(ratio * np.sqrt(to_periods / from_periods))
