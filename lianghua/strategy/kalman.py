"""卡尔曼滤波：时变对冲比估计。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_hedge(y: pd.Series, x: pd.Series,
                    q: float = 1e-4, r: float = 1e-2) -> pd.Series:
    """观测 y_t = beta_t * x_t + 噪声。返回时变 beta_t 序列。"""
    df = pd.concat([y.astype(float), x.astype(float)], axis=1).dropna()
    df.columns = ["y", "x"]
    n = len(df)
    beta = np.zeros(n)
    P = 1.0
    beta[0] = 0.0
    for t in range(1, n):
        # 预测
        P = P + q
        # 更新
        xt = df["x"].iloc[t]
        yhat = beta[t - 1] * xt
        S = xt * P * xt + r
        K = P * xt / S
        beta[t] = beta[t - 1] + K * (df["y"].iloc[t] - yhat)
        P = (1 - K * xt) * P
    return pd.Series(beta, index=df.index, name="kalman_beta")
