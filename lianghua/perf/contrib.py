"""收益贡献分解：组合日收益的时间/成分拆解。"""
from __future__ import annotations

import pandas as pd


def time_contrib(weights: pd.Series, asset_ret: pd.DataFrame) -> pd.Series:
    """各资产按权重贡献到组合日收益（横截面加总）。

    隐性修复：权重含资产不在收益表中时，reindex 会引入 NaN 列污染加总；
    现对齐后用 0 填充（无数据资产视为零贡献），并对空交集给出清晰错误。
    """
    w = weights.astype(float)
    common = [c for c in w.index if c in asset_ret.columns]
    if not common:
        raise ValueError("weights 与 asset_ret 无共同资产，无法分解贡献")
    r = asset_ret.astype(float).reindex(columns=w.index).fillna(0.0)
    return (r * w).sum(axis=1).rename("contrib")


def cumulative_contrib(weights: pd.Series, asset_ret: pd.DataFrame) -> pd.DataFrame:
    """累计贡献（各资产对组合累计收益的拉动）。"""
    daily = time_contrib(weights, asset_ret)
    cum = (1 + asset_ret.reindex(columns=weights.index).fillna(0.0)).cumprod() - 1
    return (cum * weights).rename(columns=lambda c: f"{c}_contrib")
