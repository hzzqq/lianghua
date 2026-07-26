"""因子分层回测与 IC 衰减分析。

输入因子序列 factor、收益序列，输出分组表现、IC 衰减等结构化结果，供 UI 绘图。
全部纯 pandas/numpy，零额外依赖。

本次打磨（隐性修复）：
- ic_series 原实现对每个时点 O(n) 重算全样本 IC 再滚动平均，序列一长即 O(n^2) 性能悬崖；
  且 app.py 以 ic_series(..., lags=10) 调用，而原函数签名只有 window，会 TypeError 崩溃。
  现改为标准 trailing-window 滚动 rank-IC（O(n)），并兼容 lags 别名。
- quantile_returns 直接 qcut，因子区分度低/常数时 qcut 抛 ValueError 崩溃；现过滤非有限值、
  因子无分化时退化为单组均值，并 duplicates="drop" 防止低基数因子报错。
- long_short_by_factor 在出现 NaN 组时静默得到 NaN/错误价差；现丢弃 NaN 组、不足两组返回 NaN。
新增能力：long_short_detail 提供 top/bottom/spread/组数 的结构化可观测视图。
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


def ic_series(factor: pd.Series, returns: pd.Series, window: int = 20,
              lags: int | None = None) -> pd.Series:
    """滚动 IC（截至每期的因子与未来收益相关系数）。

    隐性修复：原实现对每个 t 用 factor.loc[:t] 全样本重算 IC 再 rolling.mean，
    复杂度为 O(n^2)，数据一长即卡死；现改为标准 trailing-window 滚动 rank-IC（O(n)）。
    同时兼容历史调用 ic_series(..., lags=K)（lags 作为 window 的别名）。
    """
    if lags is not None:
        window = lags
    if window < 2:
        raise ValueError("window 必须 >= 2")
    fr = pd.Series(factor).rank()
    rr = pd.Series(returns).rank()
    df = pd.concat([fr, rr], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < window:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1]).dropna()


def ic_decay(factor: pd.Series, returns: pd.Series, lags=(1, 5, 10, 20)) -> pd.Series:
    """IC 随预测周期衰减。"""
    res = {}
    for L in lags:
        fwd = returns.shift(-L)
        res[L] = _ic(factor, fwd)
    return pd.Series(res)


def quantile_returns(factor: pd.Series, returns: pd.Series, n: int = 5) -> pd.Series:
    """按因子值分 n 组，返回每组平均收益。

    隐性修复：原实现 pd.qcut(df['f'].rank(), n) 在因子区分度低或为常数时
    直接抛 ValueError 崩溃。现过滤非有限值、因子无分化退化为单组均值，并用
    duplicates='drop' 处理低基数因子的并列边界。
    """
    df = pd.DataFrame({"f": factor, "r": returns}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    if df["f"].nunique() < 2:
        return pd.Series({0: float(df["r"].mean())})
    try:
        df["g"] = pd.qcut(df["f"].rank(method="first"), n, labels=False, duplicates="drop")
    except ValueError:
        k = max(2, df["f"].nunique())
        df["g"] = pd.qcut(df["f"].rank(method="first"), k, labels=False, duplicates="drop")
    return df.groupby("g")["r"].mean()


def long_short_by_factor(factor: pd.Series, returns: pd.Series, n: int = 5) -> float:
    """多空组合：最高组收益 - 最低组收益。"""
    grp = quantile_returns(factor, returns, n).dropna()
    if len(grp) < 2:
        return float("nan")
    return float(grp.iloc[-1] - grp.iloc[0])


def long_short_detail(factor: pd.Series, returns: pd.Series, n: int = 5) -> dict:
    """new_requirement（新增能力）：多空组合的结构化可观测视图。

    返回 {top, bottom, spread, n_groups}：最高组/最低组收益、价差、实际组数，
    供 UI 与报告展示，并在组数不足时显式给出 NaN 而非静默错误价差。
    """
    grp = quantile_returns(factor, returns, n).dropna()
    if len(grp) < 2:
        return {"top": float("nan"), "bottom": float("nan"),
                "spread": float("nan"), "n_groups": int(len(grp))}
    return {
        "top": float(grp.iloc[-1]),
        "bottom": float(grp.iloc[0]),
        "spread": float(grp.iloc[-1] - grp.iloc[0]),
        "n_groups": int(len(grp)),
    }


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
        "long_short_detail": long_short_detail(factor, returns, n),
        "ic_decay": ic_decay(factor, returns, lags),
    }
