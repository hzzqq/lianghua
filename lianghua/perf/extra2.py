"""补充绩效比率（迭代 181+，超自驱动目标）：payoff / hit / profit_factor /
outlier / recovery。

输入为净值曲线 equity(pd.Series) 或收益序列 returns(pd.Series)。
纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ratios import _returns
from .extra import _max_drawdown


def payoff_ratio(equity) -> float:
    """盈亏比：平均盈利 / 平均亏损（绝对值）。>1 表示盈利单大于亏损单。"""
    r = _returns(equity)
    wins = r[r > 0]
    losses = -r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(wins.mean() / (losses.mean() + 1e-12))


def hit_ratio(equity) -> float:
    """命中率：盈利周期占比（胜率）。∈[0,1]。"""
    r = _returns(equity)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def profit_factor(equity) -> float:
    """盈利因子：总盈利 / 总亏损（绝对值）。>=1 策略长期可盈利。"""
    r = _returns(equity)
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg <= 1e-12:
        return 99.0 if pos > 0 else 0.0
    return float(pos / neg)


def outlier_ratio(equity, top_n: int = 5) -> float:
    """异常贡献比：收益最高的 top_n 日占全部日收益绝对值之和的比例。
    越高说明收益越依赖少数极端日（脆弱）。"""
    r = _returns(equity)
    if len(r) == 0:
        return 0.0
    top = r.nlargest(int(top_n)).sum()
    total_abs = r.abs().sum()
    return float(abs(top) / (total_abs + 1e-12))


def recovery_factor(equity) -> float:
    """恢复因子：净收益 / 最大回撤。衡量爬出坑的速度（越大越好）。"""
    eq = pd.Series(equity).astype(float)
    if len(eq) < 2:
        return 0.0
    net = eq.iloc[-1] / (eq.iloc[0] + 1e-12) - 1.0
    mdd = abs(_max_drawdown(eq))
    return float(net / (mdd + 1e-12))


__all__ = ["payoff_ratio", "hit_ratio", "profit_factor", "outlier_ratio",
           "recovery_factor"]
