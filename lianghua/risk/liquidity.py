"""流动性风险：Amihud 非流动性指标 + 换手冲击成本估计。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def amihud(returns: pd.Series, volume: pd.Series, price: pd.Series) -> float:
    """Amihud 非流动性 = |收益| / 成交额 的均值（越大越难成交）。"""
    ret = returns.replace(0, np.nan).abs()
    dv = (volume * price).replace(0, np.nan)
    illiq = (ret / dv).replace([np.inf, -np.inf], np.nan)
    return float(illiq.mean())


def liquidity_cost(weights: dict, adv: dict, participant_rate: float = 0.1) -> pd.DataFrame:
    """participant_rate：单次成交占日均成交量的比例，用于估计冲击成本。

    返回各资产的简化冲击成本（%）。
    """
    rows = []
    for a, w in weights.items():
        adv_a = adv.get(a, 0.0)
        # 占 ADV 比例越高冲击越大（线性近似，封顶 100%）
        cost = min(participant_rate, 1.0) * (0.1 if adv_a > 0 else 10.0)
        rows.append({"资产": a, "权重": w, "冲击成本%": cost * 100})
    return pd.DataFrame(rows)
