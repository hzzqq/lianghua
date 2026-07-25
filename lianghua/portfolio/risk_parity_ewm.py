"""风险平价（指数加权协方差）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewm_cov(returns: pd.DataFrame, span: int = 60) -> np.ndarray:
    """指数加权协方差矩阵。"""
    r = returns.astype(float)
    return r.ewm(span=span, adjust=False).cov().iloc[-len(r.columns):].values


def risk_parity_ewm(returns: pd.DataFrame, span: int = 60,
                       max_iter: int = 300) -> pd.Series:
    """用 EWM 协方差做等风险贡献配置。"""
    cov = ewm_cov(returns, span)
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        mrc = cov @ w
        rc = w * mrc / (w @ cov @ w)
        w = w * (rc.mean() / (rc + 1e-12))
        w = w / w.sum()
    return pd.Series(w, index=returns.columns, name="ewm_rp")
