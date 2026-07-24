"""收益贡献分解：组合日收益的时间/成分拆解。"""
from __future__ import annotations

import pandas as pd


def time_contrib(weights: pd.Series, asset_ret: pd.DataFrame) -> pd.Series:
    """各资产按权重贡献到组合日收益（横截面加总）。"""
    w = weights.astype(float)
    r = asset_ret.astype(float).reindex(columns=w.index)
    return (r * w).sum(axis=1).rename("contrib")


def cumulative_contrib(weights: pd.Series, asset_ret: pd.DataFrame) -> pd.DataFrame:
    """累计贡献（各资产对组合累计收益的拉动）。"""
    daily = time_contrib(weights, asset_ret)
    cum = (1 + asset_ret.reindex(columns=weights.index)).cumprod() - 1
    return (cum * weights).rename(columns=lambda c: f"{c}_contrib")
