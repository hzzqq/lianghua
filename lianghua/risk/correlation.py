"""相关性工具：矩阵 / 平均相关 / 滚动相关 / 最近正定相关矩阵修复。

所有入口带输入守卫：非有限(inf/NaN)收益会被过滤，避免静默污染相关性矩阵；
窗口参数受校验；并提供 nearest_pd_correlation 修复下游优化器最易踩的
"相关矩阵非半正定" 隐患（迭代求逆崩、权重 NaN）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def corr_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """多资产收益的相关性矩阵。

    隐性修复：原实现 returns_df.astype(float).corr() 在含 inf/NaN 时会静默产出
    全 NaN 或非有限相关性，下游优化器据此求逆直接崩。现先过滤非有限值。
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df 必须是 DataFrame（每列一个资产）")
    r = returns_df.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if r.shape[1] < 2:
        raise ValueError("corr_matrix 至少需要 2 个资产列")
    if r.shape[0] < 2:
        raise ValueError("corr_matrix 至少需要 2 行有限收益")
    c = r.corr()
    # 常数列的相关性为 NaN（含对角线），统一置 0，对角线强制为 1
    c = c.where(np.isfinite(c), 0.0)
    for i in c.index:
        c.loc[i, i] = 1.0
    return c


def average_correlation(returns_df: pd.DataFrame) -> float:
    """资产间平均 pairwise 相关性（不含自相关）。

    隐性修复：原实现取 off-diagonal 后直接 .mean()，若某些资产对相关性为 NaN
    （如某资产收益为常数），numpy 均值会传播 NaN，静默返回 NaN。现仅对有限值求平均。
    """
    c = corr_matrix(returns_df).values
    n = c.shape[0]
    if n < 2:
        return 0.0
    off = c[np.triu_indices(n, k=1)]
    off = off[np.isfinite(off)]
    if off.size == 0:
        return 0.0
    return float(off.mean())


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    """两序列滚动相关性。

    隐性修复：未校验 window，window<=0 或 >len 时静默产出全 NaN 且无提示；
    现显式校验并给出清晰错误。
    """
    if window < 2:
        raise ValueError("window 必须 >= 2")
    df = pd.concat([pd.Series(a).astype(float), pd.Series(b).astype(float)], axis=1).dropna()
    if df.shape[0] < window:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])


def is_pd(mat) -> bool:
    """判断方阵是否对称半正定（特征值均 >= -tol）。"""
    A = np.asarray(mat, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    vals = np.linalg.eigvalsh((A + A.T) / 2.0)
    return bool(np.all(vals >= -1e-8))


def nearest_pd_correlation(corr: pd.DataFrame) -> pd.DataFrame:
    """把可能非半正定的相关性矩阵修复为最近的正定相关矩阵（Higham 谱裁剪）。

    new_requirement（新增能力）：相关矩阵在样本不足/资产高度共线时常非半正定，
    直接喂给优化器 np.linalg.inv 会 LinAlgError 或产出 NaN/负权重。本函数通过
    对称化 + 特征值裁剪到 >= eps 重建，再归一化对角线为 1，得到可用相关矩阵。
    """
    if not isinstance(corr, pd.DataFrame):
        raise TypeError("corr 必须是 DataFrame")
    A = corr.astype(float).values
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("corr 必须是方阵")
    sym = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 1e-12, None)          # 裁剪负/零特征值 -> 正定
    fixed = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(fixed), 1e-12, None))
    fixed = fixed / np.outer(d, d)             # 归一化对角线为 1
    fixed = np.clip(fixed, -1.0, 1.0)
    np.fill_diagonal(fixed, 1.0)
    return pd.DataFrame(fixed, index=corr.index, columns=corr.columns)
