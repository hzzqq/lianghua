"""最小方差组合（全局最小风险组合），含数值稳定化。

v2 改进（polish cycle39）：
- 输入校验：cov 必须为有限、方阵、且至少 1×1；否则抛清晰 ValueError，
  杜绝隐性 NaN/崩溃。
- 原实现用裸 ``np.linalg.inv(c)``：协方差奇异（高度共线资产）或含非有限值时
  会抛 LinAlgError / 产出 NaN 权重。现改用 ``np.linalg.pinv`` + 对角 ridge
  正则（shrink 向单位阵收缩），既保证数值稳定，又让权重平滑趋向等权而非发散。
- 新增 ``min_variance_from_returns(returns, ridge)``：直接从收益矩阵构建（去均值、
  剔非有限、带对角收缩的）协方差并求最小方差权重，省去调用方手写协方差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_cov(cov: np.ndarray) -> np.ndarray:
    """校验协方差矩阵有限、方阵、非空，返回 float 副本。"""
    c = np.array(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov 必须是 n×n 方阵")
    n = c.shape[0]
    if n == 0:
        raise ValueError("cov 不能为空（至少 1×1）")
    if not np.all(np.isfinite(c)):
        raise ValueError("cov 含 NaN/inf，无法求最小方差权重")
    return c, n


def min_variance(cov: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """全局最小方差权重（非负归一化）。

    ridge: 对角正则强度（>=0）。默认极小仅保底，避免奇异矩阵求逆崩溃；
           调大则权重向等权平滑收缩，可抑制估计误差带来的极端权重。
    返回 n 维权重（和为 1）。cov 为 1×1 时直接返回 [1.0]。
    """
    c, n = _validate_cov(cov)
    if n == 1:
        return np.array([1.0], dtype=float)
    s = float(np.clip(ridge, 0.0, None))
    # pinv 对奇异/病态矩阵稳健；ridge 进一步保证良态
    inv = np.linalg.pinv(c + s * np.eye(n))
    ones = np.ones(n)
    denom = float(ones @ inv @ ones)
    if not np.isfinite(denom) or abs(denom) <= 0:
        return np.full(n, 1.0 / n)
    w = inv @ ones / denom
    w = w / w.sum()
    return w


def min_variance_from_returns(returns: pd.DataFrame | np.ndarray,
                              ridge: float = 0.1) -> np.ndarray:
    """从收益矩阵直接求最小方差权重（新增端到端能力）。

    returns: T×n 收益矩阵（DataFrame 每列一个资产 / ndarray）。
    自动去均值、剔除非有限行，并对样本协方差施加对角收缩（shrink=ridge），
    解决样本不足 / 高度共线时协方差奇异产生 NaN 权重的隐性问题。

    返回 n 维权重向量。资产数为 0 或不足 2 期时安全返回等权。
    """
    r = np.array(returns, dtype=float) if not isinstance(returns, pd.DataFrame) \
        else returns.to_numpy(dtype=float)
    if r.ndim != 2:
        raise ValueError("returns 必须是 T×n 的二维收益矩阵")
    r = r[np.isfinite(r).all(axis=1)]
    n = r.shape[1] if r.ndim == 2 else 0
    if n <= 0:
        return np.array([], dtype=float)
    if r.shape[0] < 2 or n == 1:
        return np.full(n, 1.0 / n)
    r = r - r.mean(axis=0, keepdims=True)
    cov = (r.T @ r) / max(1, r.shape[0] - 1)
    s = float(np.clip(ridge, 0.0, 1.0))
    # 对角收缩：cov_s = (1-s)*cov + s*mean(diag)*I
    mean_var = float(np.mean(np.diag(cov)))
    if not np.isfinite(mean_var) or mean_var <= 0:
        mean_var = 1.0
    shrunk = (1.0 - s) * cov + s * np.eye(n) * mean_var
    return min_variance(shrunk, ridge=1e-10)
