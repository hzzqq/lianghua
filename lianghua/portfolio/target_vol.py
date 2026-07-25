"""目标波动率组合：把组合波动缩放到目标水平。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def target_vol_weights(asset_returns: pd.DataFrame, target_vol: float = 0.12,
                        periods: int = 252) -> pd.Series:
    """在等权基础上，按总波动反推使组合波动≈target_vol 的权重。"""
    r = asset_returns.astype(float).dropna()
    if r.empty:
        return pd.Series(dtype=float)
    w0 = pd.Series(1.0 / r.shape[1], index=r.columns)
    port_vol = np.sqrt(w0 @ r.cov().values @ w0) * np.sqrt(periods)
    if port_vol == 0:
        return w0
    scale = target_vol / port_vol
    return (w0 * scale).rename("weight")
