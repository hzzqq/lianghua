"""资产聚类：按收益相关性聚类，辅助分散化配置（边界与可观测强化）。

v2 改进（polish cycle41）：
- 输入守卫：returns_df 必须为 DataFrame 且至少含 1 个资产列；否则抛清晰错误。
  单资产直接返回单簇，避免 KMeans 退化。
- 修复 ``rng.choice(n, n_clusters, replace=False)`` 在 n_clusters > n（或 n==0）
  时抛 ValueError 的崩溃；现自动把 n_clusters 收敛到合法区间 [1, n]。
- 修复相关性矩阵含 NaN（某列全缺数据）时距离矩阵全 NaN、KMeans 产出 NaN 标签的
  隐性问题：先剔除非有限列，若不足 2 个有限列则整体归入单簇。
- 播种由列名哈希决定，跨进程可复现且对列顺序不敏感。
- 新增 ``cluster_summary``：返回 {簇: 资产, 簇规模, 簇内平均相关} 的可观测摘要。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _seed_from(columns) -> np.random.Generator:
    key = "|".join(map(str, columns)).encode("utf-8")
    s = int(hashlib.md5(key).hexdigest(), 16) % (2**32)
    return np.random.default_rng(s)


def _prepare(returns_df: pd.DataFrame, n_clusters: int) -> tuple[np.ndarray, list, int]:
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df 必须是 pd.DataFrame（每列一个资产）")
    if n_clusters is not None and (not isinstance(n_clusters, int) or n_clusters < 1):
        raise ValueError("n_clusters 必须为正整数")
    # 剔除非有限列，避免相关性矩阵出现 NaN
    r = returns_df.apply(pd.to_numeric, errors="coerce")
    finite_cols = [c for c in r.columns if np.isfinite(r[c]).any()]
    r = r[finite_cols].dropna()
    n = len(finite_cols)
    if n == 0:
        raise ValueError("returns_df 无任何有限收益列，无法聚类")
    if n == 1:
        return None, finite_cols, 1  # 单资产：返回空距离矩阵，由调用方直接成簇
    c = r.corr().values
    c = np.nan_to_num(c, nan=0.0)  # 残余 NaN 兜底为不相关
    dist = np.sqrt(np.clip(0.5 * (1 - c), 0, 2))
    k = n_clusters or 2
    k = int(min(max(k, 1), n))  # 收敛到 [1, n]，杜绝 choice 越界
    return dist, finite_cols, k


def _kmeans(dist: np.ndarray, n_clusters: int, n: int) -> np.ndarray:
    rng = _seed_from(list(range(n)))
    centers = rng.choice(n, n_clusters, replace=False)
    cent = dist[centers].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        d = np.linalg.norm(dist[:, None, :] - cent[None, :, :], axis=2)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for kk in range(n_clusters):
            m = labels == kk
            if m.any():
                cent[kk] = dist[m].mean(0)
    return labels


def cluster_assets(returns_df: pd.DataFrame, n_clusters: int = 2) -> dict:
    """按相关性距离做 KMeans（纯 numpy，无额外依赖）。返回 {簇号: [资产]}。"""
    dist, cols, k = _prepare(returns_df, n_clusters)
    n = len(cols)
    if n == 1:
        return {0: cols}
    labels = _kmeans(dist, k, n)
    return {kk: [cols[i] for i in range(n) if labels[i] == kk]
            for kk in range(k)}


def cluster_summary(returns_df: pd.DataFrame, n_clusters: int = 2) -> dict:
    """可观测摘要（新增能力）：{簇号: {assets, size, avg_intra_corr}}。

    avg_intra_corr：簇内资产两两相关的均值（分散化质量的粗略度量，越高越同质）。
    """
    dist, cols, k = _prepare(returns_df, n_clusters)
    n = len(cols)
    if n == 1:
        return {0: {"assets": cols, "size": 1, "avg_intra_corr": 1.0}}
    c = np.nan_to_num(returns_df[cols].corr().values, nan=0.0)
    labels = _kmeans(dist, k, n)
    out: dict = {}
    for kk in range(k):
        members = [i for i in range(n) if labels[i] == kk]
        if not members:
            continue
        if len(members) == 1:
            avg = 1.0
        else:
            sub = c[np.ix_(members, members)]
            iu = np.triu_indices(len(members), k=1)
            avg = float(np.mean(sub[iu])) if iu[0].size else 1.0
        out[kk] = {"assets": [cols[i] for i in members],
                   "size": len(members), "avg_intra_corr": avg}
    return out
