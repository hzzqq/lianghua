"""因子预处理：缩尾 / 标准化 / 行业中性化（鲁棒化 + 能力扩展）。

v2 改进（polish cycle43）：
- winsorize 新增输入守卫：lower<upper 且均落在 (0,1)；空 / 全 NaN 序列不再因
  quantile 返回 NaN 而产出 NaN 边界；lower>=upper 时自动交换，避免 clip 出错。
- zscore 修复零标准差（常数列）被 1e-12 除出巨大值的隐性 bug：std 趋零 /
  非有限时直接返回全 0，而非伪造噪声；同时对非有限输入返回全 NaN 而非崩溃。
- 新增 robust_zscore：基于中位数 + MAD 的稳健标准化（因子研究标配，对离群值不敏感）。
- 新增 winsorize_group：按分组（如行业）分别缩尾，避免跨组分布差异导致的失真。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_quantiles(lower: float, upper: float) -> tuple[float, float]:
    if not (0.0 < lower < 1.0 and 0.0 < upper < 1.0):
        raise ValueError("lower/upper 必须落在 (0, 1)")
    if lower > upper:
        lower, upper = upper, lower  # 自动交换，避免 clip 边界倒置
    return lower, upper


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """按分位缩尾，抑制极端值。"""
    lower, upper = _validate_quantiles(lower, upper)
    s = series.astype(float)
    if len(s) == 0 or not np.isfinite(s.to_numpy()).any():
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return s  # 全 NaN 等退化情形：原样返回
    return s.clip(lo, hi)


def zscore(series: pd.Series) -> pd.Series:
    """标准化（去均值 / 单位标准差）。零标准差或非有限输入稳健处理。"""
    s = series.astype(float)
    std = s.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=s.index, name="z")
    return ((s - s.mean()) / std).rename("z")


def robust_zscore(series: pd.Series) -> pd.Series:
    """稳健标准化（新增能力）：基于中位数与 MAD（中位绝对偏差）。

    z = (x - median) / (1.4826 * MAD)，对离群值远较普通 zscore 稳健，
    是因子研究中的标准做法。MAD 为 0（半数以上取值相同）时返回全 0。
    """
    s = series.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(0.0, index=s.index, name="rz")
    return ((s - med) / scale).rename("rz")


def winsorize_group(series: pd.Series, group: pd.Series,
                    lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """按分组（如行业）分别缩尾（新增能力）。

    跨组分布差异大时，全局缩尾会扭曲组内排序；分组缩尾保证每组内部可比。
    返回与 series 等长的缩尾序列。
    """
    _validate_quantiles(lower, upper)
    s = series.astype(float)
    g = group.astype(str)
    if len(s) == 0:
        return s
    return s.groupby(g).transform(lambda x: winsorize(x, lower, upper))


def neutralize(factor: pd.Series, industry: pd.Series) -> pd.Series:
    """行业中性化：对行业哑变量回归取残差。"""
    f = factor.astype(float)
    ind_series = industry.astype(str).rename("ind")
    dummies = pd.get_dummies(pd.Categorical(ind_series))
    X = pd.concat([pd.Series(1.0, index=f.index, name="const"), dummies], axis=1).astype(float)
    y = f.values
    Xv = X.values
    try:
        beta = np.linalg.lstsq(Xv, y, rcond=None)[0]
        resid = y - Xv @ beta
    except Exception:
        resid = y - y.mean()
    return pd.Series(resid, index=f.index, name="neutral")
