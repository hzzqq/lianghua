"""日内策略基类（基于分钟线 bar）。

输入 df 需含 open/high/low/close（分钟级），index 为带时间的 datetime。
generate_signals 输出与 df 同长的信号序列：1=做多, -1=做空, 0=观望。
可用 make_demo_minutes() 生成合成分钟数据用于测试。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class IntradayStrategy:
    name = "intraday_base"
    description = "日内策略基类"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class IntradayBreakout(IntradayStrategy):
    """开盘区间突破：以开盘后 N 分钟的高低区间为基准，突破上沿做多、下沿做空。"""

    name = "intraday_breakout"
    description = "日内开盘区间突破"

    def __init__(self, lookback: int = 30, hold: int = 60):
        self.lookback = lookback
        self.hold = hold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # 突破前 lookback 根收盘价的最高/最低（不含当前 bar），随机游走下可频繁触发
        ref_hi = df["close"].rolling(self.lookback, min_periods=2).max().shift(1)
        ref_lo = df["close"].rolling(self.lookback, min_periods=2).min().shift(1)
        close = df["close"]
        sig = pd.Series(0, index=df.index)
        sig[close > ref_hi] = 1
        sig[close < ref_lo] = -1
        return sig


class IntradayMeanReversion(IntradayStrategy):
    """日内均值回复：价格偏离日内 VWAP 超过 k 倍标准差则反向。"""

    name = "intraday_mean_reversion"
    description = "日内 VWAP 均值回复"

    def __init__(self, window: int = 60, k: float = 2.0):
        self.window = window
        self.k = k

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vwap = typical.rolling(self.window).mean()
        sd = typical.rolling(self.window).std()
        z = (typical - vwap) / sd.replace(0, np.nan)
        sig = pd.Series(0, index=df.index)
        sig[z > self.k] = -1
        sig[z < -self.k] = 1
        return sig


def make_demo_minutes(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """生成合成分钟 K 线（随机游走 + 轻微日内趋势），用于测试。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="min")
    drift = np.linspace(0, 0.01, n)
    ret = rng.normal(0, 0.001, n).cumsum() * 0.01 + drift
    close = 100 * np.exp(ret)
    high = close * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
