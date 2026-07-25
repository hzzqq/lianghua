"""滚动绩效指标（迭代23）。

对净值/收益序列做滚动窗口分析，返回时间序列，便于 UI 绘图：
- rolling_sharpe      滚动夏普
- rolling_volatility  滚动年化波动率
- rolling_return      滚动年化收益
- rolling_max_drawdown 滚动最大回撤
- rolling_report      汇总 DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rolling_sharpe", "rolling_volatility", "rolling_return",
           "rolling_max_drawdown", "rolling_report"]

TRADING_DAYS = 252


def _to_returns(series: pd.Series, is_equity: bool) -> pd.Series:
    s = pd.Series(series).dropna()
    return s.pct_change().dropna() if is_equity else s


def rolling_sharpe(series: pd.Series, window: int = 60,
                   rf: float = 0.0, is_equity: bool = True) -> pd.Series:
    """滚动年化夏普比率。"""
    r = _to_returns(series, is_equity) - rf / TRADING_DAYS
    mean = r.rolling(window).mean()
    std = r.rolling(window).std()
    return (mean / std.replace(0, np.nan) * np.sqrt(TRADING_DAYS)).rename("rolling_sharpe")


def rolling_volatility(series: pd.Series, window: int = 60,
                       is_equity: bool = True) -> pd.Series:
    """滚动年化波动率。"""
    r = _to_returns(series, is_equity)
    return (r.rolling(window).std() * np.sqrt(TRADING_DAYS)).rename("rolling_vol")


def rolling_return(series: pd.Series, window: int = 60,
                   is_equity: bool = True) -> pd.Series:
    """滚动年化收益率（几何）。"""
    r = _to_returns(series, is_equity)
    def _ann(x):
        total = float(np.prod(1 + x))
        return total ** (TRADING_DAYS / len(x)) - 1
    return r.rolling(window).apply(_ann, raw=True).rename("rolling_return")


def rolling_max_drawdown(equity: pd.Series, window: int = 60) -> pd.Series:
    """滚动窗口内的最大回撤（正数表示回撤幅度）。"""
    eq = pd.Series(equity).dropna()
    def _mdd(x):
        peak = np.maximum.accumulate(x)
        return float(((peak - x) / peak).max())
    return eq.rolling(window).apply(_mdd, raw=True).rename("rolling_mdd")


def rolling_report(equity: pd.Series, window: int = 60) -> pd.DataFrame:
    """汇总滚动指标：夏普 / 波动 / 年化收益 / 最大回撤。"""
    eq = pd.Series(equity).dropna()
    return pd.DataFrame({
        "滚动夏普": rolling_sharpe(eq, window),
        "滚动波动率": rolling_volatility(eq, window),
        "滚动年化收益": rolling_return(eq, window),
        "滚动最大回撤": rolling_max_drawdown(eq, window),
    })
