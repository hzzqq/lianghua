"""Bootstrap：指标置信区间。"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["bootstrap_metric", "bootstrap_ci"]


def bootstrap_metric(equity, metric_fn, n_boot: int = 500, seed: int = 0,
                     ci: tuple = (2.5, 97.5), return_dist: bool = False) -> dict:
    """对收益序列自助重采样，返回 metric_fn 的均值/分位区间。

    新增：自定义置信度 ci、median/std、有效样本数 n_valid、
    metric_fn 失败计数 n_fail（逐样本失败隔离）、可选返回分布。
    """
    if not isinstance(equity, pd.Series):
        equity = pd.Series(np.asarray(equity, dtype=float))
    vals_arr = equity.to_numpy(dtype=float)
    if not np.all(np.isfinite(vals_arr)):
        raise ValueError("equity 含非有限值(NaN/inf)，无法做 bootstrap")
    if len(vals_arr) < 2:
        raise ValueError("equity 至少需要 2 个观测点才能计算收益序列")
    r = pd.Series(vals_arr).pct_change().dropna().to_numpy()
    if len(r) == 0:
        raise ValueError("收益序列为空，无法做 bootstrap")
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f"n_boot 必须为正整数，收到 {n_boot!r}")
    if not (isinstance(ci, (tuple, list)) and len(ci) == 2 and
            0 <= ci[0] <= 100 and 0 <= ci[1] <= 100 and ci[0] <= ci[1]):
        raise ValueError(f"ci 必须为 [0,100] 内且 low<=high 的 (low, high)，收到 {ci!r}")

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    n_fail = 0
    for i in range(n_boot):
        idx = rng.integers(0, len(r), len(r))
        sample = (1 + pd.Series(r[idx])).cumprod()
        try:
            vals[i] = float(metric_fn(sample))
        except Exception:
            # 逐样本失败隔离：单样本异常不得中止整轮 bootstrap
            vals[i] = np.nan
            n_fail += 1
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        raise RuntimeError("所有 bootstrap 样本在 metric_fn 下都失败")

    lo, hi = float(ci[0]), float(ci[1])
    out = {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "low_95": float(np.percentile(finite, 2.5)),
        "high_95": float(np.percentile(finite, 97.5)),
        "low": float(np.percentile(finite, lo)),
        "high": float(np.percentile(finite, hi)),
        "ci": [lo, hi],
        "n": n_boot,
        "n_valid": int(len(finite)),
        "n_fail": int(n_fail),
    }
    if return_dist:
        out["distribution"] = finite
    return out


def bootstrap_ci(equity, metric_fn, alpha: float = 0.05, n_boot: int = 500,
                 seed: int = 0) -> tuple:
    """返回 (low, high) 双侧置信区间（便捷封装）。"""
    if not (0 < alpha < 1):
        raise ValueError(f"alpha 必须在 (0,1)，收到 {alpha!r}")
    lo = 100.0 * alpha / 2.0
    hi = 100.0 - lo
    res = bootstrap_metric(equity, metric_fn, n_boot=n_boot, seed=seed, ci=(lo, hi))
    return (res["low"], res["high"])
