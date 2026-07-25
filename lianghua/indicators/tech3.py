"""进阶技术指标（迭代 151–157）：
keltner_channels / donchian_channel / hull_moving_average / true_strength_index /
chandelier_exit / zscore / ease_of_movement / mass_index。

输入 OHLCV DataFrame 或 close Series，输出等长 Series / DataFrame。
纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wma(s, n: int) -> pd.Series:
    """加权移动平均（线性权重 1..n），窗口固定为 n。"""
    s = pd.Series(s).astype(float)
    if n < 1:
        n = 1
    return s.rolling(n).apply(
        lambda x: np.dot(x, np.arange(1, n + 1)) / (n * (n + 1) / 2), raw=True
    )


def keltner_channels(df, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """肯特纳通道：中轨=EMA(close)，上下轨=中轨±mult×ATR。"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    middle = close.ewm(span=period, adjust=False).mean()
    prev = close.shift()
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    upper = middle + mult * atr
    lower = middle - mult * atr
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})


def donchian_channel(df, period: int = 20) -> pd.DataFrame:
    """唐奇安通道：上轨=N 日最高，下轨=N 日最低，中轨=均值。"""
    high = df["high"]
    low = df["low"]
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "lower": lower, "middle": middle})


def hull_moving_average(close, period: int = 20) -> pd.Series:
    """赫尔移动平均：对 WMA 差再做短窗 WMA，更跟势且滞后更小。"""
    close = pd.Series(close).astype(float)
    half = max(period // 2, 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    wma_half = _wma(close, half)
    wma_full = _wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = _wma(diff, sqrt_p)
    return hma.rename("hma")


def true_strength_index(close, r: int = 25, s: int = 13) -> pd.Series:
    """真实强弱指数(TSI)：双平滑动量 / 双平滑绝对动量，∈(-100,100)。"""
    close = pd.Series(close).astype(float)
    m = close.diff()
    absm = m.abs()
    ema1 = m.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    a1 = absm.ewm(span=r, adjust=False).mean()
    a2 = a1.ewm(span=s, adjust=False).mean()
    tsi = 100 * ema2 / (a2 + 1e-12)
    return tsi.rename("tsi")


def chandelier_exit(df, period: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """吊灯止损：多头退场=近 N 高 - mult×ATR；空头退场=近 N 低 + mult×ATR。"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev = close.shift()
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    long_exit = highest - mult * atr
    short_exit = lowest + mult * atr
    return pd.DataFrame({"long": long_exit, "short": short_exit})


def zscore(close, period: int = 20) -> pd.Series:
    """滚动 Z 分数：价格相对其 N 日均值的标准化偏离。"""
    close = pd.Series(close).astype(float)
    mu = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return ((close - mu) / (sd + 1e-12)).rename("zscore")


def ease_of_movement(df, period: int = 14) -> pd.Series:
    """简易波动指标(EMV)：价格区间中点变动 / 成交量箱宽，平滑后反映量价推力。"""
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    prev_h = high.shift()
    prev_l = low.shift()
    box = (high + low) / 2 - (prev_h + prev_l) / 2
    box_width = (high - low) / (vol.replace(0, np.nan) + 1e-12)
    emv = box / (box_width + 1e-12)
    return emv.rolling(period).mean().rename("emv")


def mass_index(df, period: int = 9, sum_period: int = 25) -> pd.Series:
    """质量指数：双平滑(高-低)之比的累计和，用于预警趋势反转。"""
    high = df["high"]
    low = df["low"]
    e = (high - low)
    ema9 = e.ewm(span=period, adjust=False).mean()
    ema9_2 = ema9.ewm(span=period, adjust=False).mean()
    ratio = ema9 / (ema9_2 + 1e-12)
    return ratio.rolling(sum_period).sum().rename("mass_index")
