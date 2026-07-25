"""排序信号聚合：多因子分位加权组合。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_signal(signals: dict, weights: dict | None = None) -> pd.Series:
    """每个信号先转横截面分位(0~1)，再加权求和 → 复合分位信号。"""
    if not signals:
        return pd.Series(dtype=float, name="rank_score")
    mat = pd.DataFrame({k: pd.Series(v).astype(float) for k, v in signals.items()})
    rank = mat.rank(pct=True)
    w = {k: 1.0 / len(signals) for k in signals}
    if weights:
        tot = sum(weights.get(k, 0.0) for k in signals) or 1.0
        w = {k: weights.get(k, 0.0) / tot for k in signals}
    return sum(rank[k] * w[k] for k in signals).rename("rank_score")


def rank_to_position(score: pd.Series, long_th: float = 0.7,
                    short_th: float = 0.3) -> pd.Series:
    """分位 > long_th 多(1)，< short_th 空(-1)，否则 0。"""
    return pd.Series(np.where(score > long_th, 1,
                           np.where(score < short_th, -1, 0)),
                      index=score.index, name="rank_pos")
