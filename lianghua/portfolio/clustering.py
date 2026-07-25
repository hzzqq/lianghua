"""资产聚类：按收益相关性聚类，辅助分散化配置。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_assets(returns_df: pd.DataFrame, n_clusters: int = 2) -> dict:
    """按相关性距离做 KMeans（纯 numpy，无额外依赖）。返回 {簇号: [资产]}。"""
    r = returns_df.astype(float).dropna()
    c = r.corr().values
    # 距离 = sqrt(0.5*(1-corr))
    dist = np.sqrt(np.clip(0.5 * (1 - c), 0, 2))
    n = dist.shape[0]
    rng = np.random.default_rng(0)
    centers = rng.choice(n, n_clusters, replace=False)
    cent = dist[centers].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        d = np.linalg.norm(dist[:, None, :] - cent[None, :, :], axis=2)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in range(n_clusters):
            m = labels == k
            if m.any():
                cent[k] = dist[m].mean(0)
    return {k: [returns_df.columns[i] for i in range(n) if labels[i] == k]
            for k in range(n_clusters)}
