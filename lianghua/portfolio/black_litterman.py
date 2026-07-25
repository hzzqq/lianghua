"""简化 Black-Litterman：由先验收益 + 主观观点得到后验预期收益。"""
from __future__ import annotations

import numpy as np


def black_litterman(pi: np.ndarray, cov: np.ndarray, P: np.ndarray, Q: np.ndarray,
                    tau: float = 0.05, omega: np.ndarray | None = None) -> np.ndarray:
    """pi: 先验收益(n,)；cov: 协方差(n,n)；P: 观点矩阵(k,n)；Q: 观点收益(k,)。

    返回后验预期收益(n,)。
    """
    pi = np.array(pi, float)
    cov = np.array(cov, float)
    P = np.array(P, float)
    Q = np.array(Q, float)
    n = len(pi)
    tau_cov = tau * cov
    if omega is None:
        omega = 0.05 * np.eye(len(Q))
    A = np.linalg.inv(tau_cov) @ P.T @ np.linalg.inv(P @ tau_cov @ P.T + omega)
    post_pi = pi + tau_cov @ A @ (Q - P @ pi)
    return post_pi
