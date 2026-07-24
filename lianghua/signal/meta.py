"""元信号：多信号多数表决 / 加权组合。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def majority_vote(signals: dict) -> pd.Series:
    """每个时刻取多数的 −1/0/1（平票取 0）。"""
    if not signals:
        return pd.Series(dtype=float, name="meta")
    mat = pd.DataFrame({k: pd.Series(v).astype(float) for k, v in signals.items()})
    s = mat.sum(axis=1)
    out = pd.Series(0, index=mat.index)
    out[s > 0] = 1
    out[s < 0] = -1
    return out.rename("meta")


def weighted_meta(signals: dict, weights: dict | None = None) -> pd.Series:
    """加权求和后符号化（>0 多，<0 空）。"""
    if not signals:
        return pd.Series(dtype=float, name="meta_w")
    mat = pd.DataFrame({k: pd.Series(v).astype(float) for k, v in signals.items()})
    w = {k: 1.0 / len(signals) for k in signals}
    if weights:
        tot = sum(weights.get(k, 0.0) for k in signals) or 1.0
        w = {k: weights.get(k, 0.0) / tot for k in signals}
    s = sum(mat[k] * w[k] for k in signals)
    return pd.Series(np.sign(s.values), index=mat.index, name="meta_w")
