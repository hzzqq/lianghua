"""层次风险平价（HRP）：递归二分风险平价，纯 numpy，无需 scipy。

思路：按资产方差升序排序形成层次，递归地把组合按两侧子组合风险反比分配权重。

新增能力（Round 17）：
- hrp_from_returns(returns, shrink)：直接从收益矩阵构建（带收缩的）协方差并跑 HRP，
  规避原始收益高度共线时协方差奇异/非正定导致的 NaN 权重。
"""
from __future__ import annotations

import numpy as np


def hrp(cov: np.ndarray) -> np.ndarray:
    """层次风险平价权重。cov 为 n×n 协方差矩阵。

    隐性修复：
    - 校验 cov 为有限、方阵；否则抛清晰 ValueError。
    - 子组合风险 rl/rr 可能为 0（零方差资产）或负值（非正定带来数值负值）：
      分母<=0 时改为等权，sqrt 取 max(0,·)，杜绝 NaN。
    - 最终权重和为 0 时安全返回等权，避免 0/0。
    """
    c = np.array(cov, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov 必须是 n×n 方阵")
    if not np.all(np.isfinite(c)):
        raise ValueError("cov 含 NaN/inf，无法计算 HRP")
    n = c.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    var = np.diag(c).astype(float)
    order = np.argsort(var)

    def bisect(ids):
        if len(ids) <= 1:
            return np.array([1.0])
        mid = len(ids) // 2
        left, right = ids[:mid], ids[mid:]
        wl = bisect(left)
        wr = bisect(right)
        rl = float(np.sqrt(max(0.0, wl @ c[np.ix_(left, left)] @ wl))) if len(wl) > 1 else float(np.sqrt(max(0.0, var[left[0]])))
        rr = float(np.sqrt(max(0.0, wr @ c[np.ix_(right, right)] @ wr))) if len(wr) > 1 else float(np.sqrt(max(0.0, var[right[0]])))
        denom = rl + rr
        if denom <= 0:
            a = b = 0.5
        else:
            a = rr / denom
            b = rl / denom
        return np.concatenate([wl * a, wr * b])

    w = bisect(order)
    out = np.zeros(n)
    out[order] = w
    total = out.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(n, 1.0 / n)
    return out / total


def hrp_from_returns(returns: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """从收益矩阵直接求 HRP 权重（新增能力）。

    returns: T×n 收益矩阵（T 期, n 资产）。自动去均值、剔除非有限值，
    并对样本协方差施加对角收缩（shrink∈[0,1]）提升数值稳定性，
    解决高度共线/样本不足时协方差奇异产生 NaN 权重的隐性问题。

    返回 n 维权重向量。
    """
    r = np.array(returns, float)
    if r.ndim != 2:
        raise ValueError("returns 必须是 T×n 的二维收益矩阵")
    r = r[np.isfinite(r).all(axis=1)]
    if r.shape[0] < 2 or r.shape[1] < 1:
        n = r.shape[1] if r.ndim == 2 else 0
        if n <= 0:
            return np.array([], dtype=float)
        return np.full(n, 1.0 / n)
    r = r - r.mean(axis=0, keepdims=True)
    n = r.shape[1]
    # 样本协方差
    cov = (r.T @ r) / max(1, r.shape[0] - 1)
    # 对角收缩：cov_s = (1-s)*cov + s*diag(mean_var)
    s = float(np.clip(shrink, 0.0, 1.0))
    mean_var = np.mean(np.diag(cov)) if np.mean(np.diag(cov)) > 0 else 1.0
    shrunk = (1.0 - s) * cov + s * np.eye(n) * mean_var
    return hrp(shrunk)
