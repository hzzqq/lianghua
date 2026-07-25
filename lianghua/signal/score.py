"""复合信号打分：多信号加权汇总为单一分数。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def composite_score(signals: dict, weights: dict | None = None,
                     normalize: bool = True) -> pd.Series:
    """signals: {名称: 等长 Series(同index)}；weights: 各信号权重(默认等权)。

    返回加权复合分数 Series。normalize=True 时各信号先 z-score 标准化。
    """
    names = list(signals.keys())
    if not names:
        return pd.Series(dtype=float)
    mat = pd.DataFrame({k: pd.Series(v).astype(float) for k, v in signals.items()})
    if normalize:
        std = mat.std()
        # 常数信号(方差为 0)无区分度：归一化需保持为 0，而非除以 1e-12 膨胀成 ±inf 污染下游
        safe = (mat - mat.mean()) / std.where(std > 0, float("nan"))
        mat = safe.fillna(0.0)
    w = {k: 1.0 / len(names) for k in names}
    if weights:
        tot = sum(weights.get(k, 0.0) for k in names) or 1.0
        w = {k: weights.get(k, 0.0) / tot for k in names}
    return sum(mat[k] * w[k] for k in names).rename("score")
