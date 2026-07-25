"""因子收益：按因子值分组多空组合收益 / 分组单调性 / IC。

迭代强化：
- 新增 factor_group_returns：各分位组平均收益 + 多空价差（因子单调性可观测）。
- 隐性修复：原 concat 在 factor/forward_ret 长度不一致时按位置静默截断（丢数据）；
  现显式校验等长；并联合过滤非有限值（避免 NaN/inf 进 qcut/corr 静默出错）；
  <2 个有效观测直接抛错（可观测），不再返回陷阱值。
- factor_long_short 保持返回 length-1 Series（兼容既有调用 .iloc[0]）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _align_factor(factor, forward_ret) -> pd.DataFrame:
    f = pd.Series(factor).astype(float).reset_index(drop=True)
    r = pd.Series(forward_ret).astype(float).reset_index(drop=True)
    # 隐性修复：长度不一致时 concat 会按位置静默截断，掩盖数据对齐错误
    if len(f) != len(r):
        raise ValueError(
            f"factor({len(f)}) 与 forward_ret({len(r)}) 长度不一致，无法按位置对齐")
    df = pd.concat([f, r], axis=1)
    df.columns = ["f", "r"]
    # 隐性修复：dropna 仅剔 NaN，inf 仍需过滤，否则 qcut/corr 静默异常
    df = df[np.isfinite(df).all(axis=1)]
    if len(df) < 2:
        raise ValueError("有效观测不足 2 个，无法计算因子收益/IC")
    return df


def factor_long_short(factor: pd.Series, forward_ret: pd.Series,
                      n_group: int = 5) -> pd.Series:
    """按因子值分 n_group 组，第1组(低)做空、第n组(高)做多。

    返回 length-1 的 Series（均值多空收益），兼容既有 .iloc[0] 调用。
    """
    df = _align_factor(factor, forward_ret)
    df["grp"] = pd.qcut(df["f"], n_group, labels=False, duplicates="drop")
    ng = df["grp"].nunique()
    long = df.loc[df["grp"] == ng - 1, "r"].mean()
    short = df.loc[df["grp"] == 0, "r"].mean()
    ls = float(long - short)
    return pd.Series([ls], name="factor_ls")


def factor_group_returns(factor: pd.Series, forward_ret: pd.Series,
                         n_group: int = 5) -> pd.Series:
    """新增能力：各分位组平均收益 + 多空价差（long_short）。

    用于检验因子单调性：理想情况下组收益应随因子值单调上升。
    返回以分组标签(0..ng-1, 及 'long_short')为索引的 Series。
    """
    df = _align_factor(factor, forward_ret)
    df["grp"] = pd.qcut(df["f"], n_group, labels=False, duplicates="drop")
    means = df.groupby("grp")["r"].mean()
    means["long_short"] = means.iloc[-1] - means.iloc[0]
    means.name = "group_return"
    return means


def factor_ic(factor: pd.Series, forward_ret: pd.Series) -> float:
    """因子与前瞻收益的 Spearman 秩相关（IC）。"""
    df = _align_factor(factor, forward_ret)
    return float(df["f"].rank().corr(df["r"].rank()))
