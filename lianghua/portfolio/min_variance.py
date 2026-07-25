"""最小方差组合（全局最小风险组合）。"""
from __future__ import annotations

import numpy as np


def min_variance(cov: np.ndarray) -> np.ndarray:
    c = np.array(cov, float)
    n = len(c)
    inv = np.linalg.inv(c)
    ones = np.ones(n)
    w = inv @ ones / (ones @ inv @ ones)
    return w / w.sum()
