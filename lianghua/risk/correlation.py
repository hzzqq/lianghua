"""相关性工具：矩阵 / 平均相关 / 滚动相关。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def corr_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """多资产收益的相关性矩阵。"""
    return returns_df.astype(float).corr()


def average_correlation(returns_df: pd.DataFrame) -> float:
    """资产间平均 pairwise 相关性（不含自相关）。"""
    c = corr_matrix(returns_df).values
    n = c.shape[0]
    if n < 2:
        return 0.0
    off = c[np.triu_indices(n, k=1)]
    return float(off.mean())


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    """两序列滚动相关性。"""
    df = pd.concat([a.astype(float), b.astype(float)], axis=1).dropna()
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])
