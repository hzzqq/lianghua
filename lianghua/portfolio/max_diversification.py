"""最大分散化组合（最大化 Diversification Ratio）。"""
from __future__ import annotations

import numpy as np


def max_diversification(cov: np.ndarray, max_iter: int = 200,
                        lr: float = 0.05) -> np.ndarray:
    c = np.array(cov, float)
    n = len(c)
    w = np.ones(n) / n
    vol = np.sqrt(np.diag(c))
    for _ in range(max_iter):
        port_vol = np.sqrt(float(w @ c @ w))
        if port_vol <= 0:
            break
        mrc = c @ w / port_vol
        dr = float((w @ vol) / port_vol)
        grad = (vol / port_vol) - dr * (mrc / port_vol)
        w = w + lr * grad
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            w = np.ones(n) / n
        w = w / w.sum()
    return w
