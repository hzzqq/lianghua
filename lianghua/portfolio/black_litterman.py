"""简化 Black-Litterman：由先验收益 + 主观观点得到后验预期收益（数值稳定化）。

v2 改进（polish cycle40）：
- 输入校验：pi / cov / P / Q 必须有限、形状一致（cov 为 n×n，P 为 k×n，Q 为 k）；
  否则抛清晰 ValueError，杜绝隐性 NaN/崩溃。
- 原实现用裸 ``np.linalg.inv(tau_cov)`` 与 ``np.linalg.inv(P@tau_cov@P.T + omega)``：
  cov 奇异（共线资产）或 omega 病态时直接抛 LinAlgError。现改用 ``np.linalg.pinv``
  并对 tau_cov 施加微小 ridge，保证数值稳定。
- 新增 ``black_litterman_weights(...)``：端到端把后验收益映射为归一化多头权重
  （非负投影 + 归一化），可直接喂给组合构建，省去调用方手写二次规划近似。
"""
from __future__ import annotations

import numpy as np


def _check_finite(name: str, a: np.ndarray):
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 含 NaN/inf，无法计算 Black-Litterman")


def black_litterman(pi: np.ndarray, cov: np.ndarray, P: np.ndarray, Q: np.ndarray,
                    tau: float = 0.05, omega: np.ndarray | None = None,
                    ridge: float = 1e-10) -> np.ndarray:
    """pi: 先验收益(n,)；cov: 协方差(n,n)；P: 观点矩阵(k,n)；Q: 观点收益(k,)。

    返回后验预期收益(n,)。数值稳定：pinv + ridge 防奇异。
    """
    pi = np.array(pi, dtype=float)
    cov = np.array(cov, dtype=float)
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    n = len(pi)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != n:
        raise ValueError("cov 必须是 n×n 方阵且与 pi 长度一致")
    if P.ndim != 2 or P.shape[1] != n:
        raise ValueError(f"P 必须是 k×n 矩阵（n={n}），收到形状 {P.shape}")
    k = P.shape[0]
    if Q.ndim != 1 or Q.shape[0] != k:
        raise ValueError(f"Q 必须是长度为 k={k} 的向量")
    _check_finite("pi", pi)
    _check_finite("cov", cov)
    _check_finite("P", P)
    _check_finite("Q", Q)
    if not (0.0 < tau < 1.0):
        raise ValueError("tau 必须介于 (0, 1)")

    tau_cov = tau * (cov + ridge * np.eye(n))
    if omega is None:
        # 观点误差随 P@tau_cov@P.T 自洽，避免全零 omega 低估不确定性
        omega = np.diag(np.diag(P @ tau_cov @ P.T)) + ridge * np.eye(k)
    else:
        omega = np.array(omega, dtype=float)
        if omega.shape != (k, k):
            raise ValueError(f"omega 必须是 k×k（{k},{k}），收到 {omega.shape}")
        _check_finite("omega", omega)

    # pinv 对奇异/病态矩阵稳健
    A = np.linalg.pinv(tau_cov) @ P.T @ np.linalg.pinv(P @ tau_cov @ P.T + omega)
    post_pi = pi + tau_cov @ A @ (Q - P @ pi)
    return post_pi


def black_litterman_weights(pi: np.ndarray, cov: np.ndarray, P: np.ndarray,
                             Q: np.ndarray, tau: float = 0.05,
                             omega: np.ndarray | None = None,
                             ridge: float = 1e-10) -> np.ndarray:
    """端到端：Black-Litterman 后验收益 -> 归一化多头权重（新增能力）。

    采用均值-方差式非负投影（与 portfolio.optimizer._mv_weights 一致）：
    w ∝ max(0, Σ⁻¹·μ)，再归一化为和为 1。后验收益为空或全负时安全回退等权。
    """
    post_pi = black_litterman(pi, cov, P, Q, tau=tau, omega=omega, ridge=ridge)
    c = np.array(cov, dtype=float) + ridge * np.eye(len(pi))
    try:
        inv = np.linalg.pinv(c)
        raw = inv @ post_pi
    except np.linalg.LinAlgError:
        raw = np.ones_like(post_pi)
    raw = np.clip(raw, 0, None)
    if raw.sum() <= 0:
        raw = np.ones_like(post_pi)
    return raw / raw.sum()
