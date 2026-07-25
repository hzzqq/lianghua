"""因子分层回测与 IC 衰减分析。

输入因子序列 factor、收益序列，输出分组表现、IC 衰减等结构化结果，供 UI 绘图。
全部纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ic(factor: pd.Series, fwd: pd.Series) -> float:
    """全样本 IC = Pearson(rank(factor), rank(fwd))。"""
    df = pd.DataFrame({"f": factor, "r": fwd}).dropna()
    if len(df) < 10:
        return float("nan")
    return float(df["f"].rank().corr(df["r"].rank()))


def ic_series(factor: pd.Series, returns: pd.Series, window: int = 20) -> pd.Series:
    """滚动 IC（截至每期的因子与未来收益相关系数）。"""
    out = {}
    for t in factor.dropna().index:
        if t not in returns.index:
            continue
        out[t] = _ic(factor.loc[:t], returns.loc[:t])
    return pd.Series(out).rolling(window).mean().dropna()


def ic_decay(factor: pd.Series, returns: pd.Series, lags=(1, 5, 10, 20)) -> pd.Series:
    """IC 随预测周期衰减。"""
    res = {}
    for L in lags:
        fwd = returns.shift(-L)
        res[L] = _ic(factor, fwd)
    return pd.Series(res)


def quantile_returns(factor: pd.Series, returns: pd.Series, n: int = 5) -> pd.Series:
    """按因子值分 n 组，返回每组平均收益。"""
    df = pd.DataFrame({"f": factor, "r": returns}).dropna()
    df["g"] = pd.qcut(df["f"].rank(method="first"), n, labels=False)
    return df.groupby("g")["r"].mean()


def long_short_by_factor(factor: pd.Series, returns: pd.Series, n: int = 5) -> float:
    """多空组合：最高组收益 - 最低组收益。"""
    grp = quantile_returns(factor, returns, n)
    return float(grp.iloc[-1] - grp.iloc[0])


def factor_layer_report(
    factor: pd.Series,
    returns: pd.Series,
    n: int = 5,
    lags=(1, 5, 10, 20),
) -> dict:
    """汇总因子分层报告。"""
    return {
        "quantile_returns": quantile_returns(factor, returns, n),
        "long_short": long_short_by_factor(factor, returns, n),
        "ic_decay": ic_decay(factor, returns, lags),
    }
