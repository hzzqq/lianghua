"""多策略合成回测：先各策略生成信号，再按权重合成仓位回测。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lianghua.backtest.vectorized import vectorized_backtest


def multi_strategy_backtest(df: pd.DataFrame, signals: dict,
                            weights: dict | None = None, init_cash: float = 1_000_000.0,
                            cost_rate: float = 0.0005) -> dict:
    """signals: {名称: 与 df 同 index 的仓位 Series(如 -1/0/1)}。

    按权重合成综合仓位（clip 到 [-1,1]）后向量化回测。
    """
    names = list(signals.keys())
    if not names:
        return {}
    mat = pd.DataFrame({k: pd.Series(v).astype(float) for k, v in signals.items()})
    w = {k: 1.0 / len(names) for k in names}
    if weights:
        tot = sum(weights.get(k, 0.0) for k in names) or 1.0
        w = {k: weights.get(k, 0.0) / tot for k in names}
    composite = sum(mat[k] * w[k] for k in names).clip(-1, 1)
    return vectorized_backtest(df, composite, init_cash=init_cash, cost_rate=cost_rate)
