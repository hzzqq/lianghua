"""Beta / Alpha（CAPM）风险度量。

迭代强化：
- 新增 rolling_beta（滚动 Beta，风险暴露时变可观测）。
- 隐性修复：原仅 dropna 漏掉 inf，会静默污染 np.cov 产出 inf/nan Beta；
  现对两序列联合过滤非有限值后再回归。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _align(asset_ret, bench_ret) -> pd.DataFrame:
    a = pd.Series(asset_ret).astype(float)
    b = pd.Series(bench_ret).astype(float)
    df = pd.concat([a, b], axis=1)
    df.columns = ["asset", "bench"]
    # 隐性修复：dropna 仅剔 NaN；inf 仍需显式过滤，否则 cov 静默变 inf
    df = df[np.isfinite(df).all(axis=1)]
    return df


def beta(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    """资产对基准的 Beta（联合过滤非有限值后线性回归斜率）。"""
    df = _align(asset_ret, bench_ret)
    if len(df) < 2:
        return 0.0
    cov = np.cov(df["asset"].values, df["bench"].values)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0


def alpha_annual(asset_ret: pd.Series, bench_ret: pd.Series,
                 rf: float = 0.0, periods: int = 252) -> float:
    """年化 Jensen's Alpha（扣除无风险与 Beta 暴露后）。"""
    df = _align(asset_ret, bench_ret)
    if len(df) < 2:
        return 0.0
    beta_ = beta(df["asset"], df["bench"])
    excess = (df["asset"] - rf / periods) - beta_ * (df["bench"] - rf / periods)
    return float(excess.mean() * periods)


def capm_residuals(asset_ret: pd.Series, bench_ret: pd.Series) -> pd.Series:
    """回归残差序列（特异性收益）。"""
    df = _align(asset_ret, bench_ret)
    beta_ = beta(df["asset"], df["bench"])
    return (df["asset"] - beta_ * df["bench"]).rename("residual")


def rolling_beta(asset_ret: pd.Series, bench_ret: pd.Series,
                 window: int = 30, min_periods: int = 2) -> pd.Series:
    """滚动 Beta（新增能力）：逐窗口计算资产对基准的 Beta。

    返回与 asset_ret 对齐的 Series；样本不足 min_periods 的窗口为 NaN。
    隐性修复：窗口内非有限值导致 cov 失效时该窗返回 NaN（可观测），而非污染整体。
    """
    if window < 1:
        raise ValueError("window 必须 >= 1")
    df = _align(asset_ret, bench_ret)
    if df.empty:
        return pd.Series(dtype=float)
    idx = df.index
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = df.iloc[lo:i + 1]
        if len(seg) < min_periods:
            continue
        cov = np.cov(seg["asset"].values, seg["bench"].values)
        out[i] = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
    return pd.Series(out, index=idx, name="rolling_beta")
