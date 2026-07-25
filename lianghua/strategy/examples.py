"""示例策略库：双均线 / MACD / 动量。

均继承 StrategyBase，可直接被回测引擎加载。
"""
from __future__ import annotations

import pandas as pd

from .base import StrategyBase


class SMACrossStrategy(StrategyBase):
    """双均线交叉：短均上穿长均买入，下穿卖出。"""

    name = "sma_cross"
    description = "双均线交叉（默认 5/20 日）"

    def __init__(self, fast: int = 5, slow: int = 20):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ma_f = self.sma(close, self.fast)
        ma_s = self.sma(close, self.slow)
        diff = ma_f - ma_s
        signal = pd.Series(0, index=df.index)
        signal[diff > 0] = 1
        signal[diff < 0] = -1
        # 仅在穿越时给信号，避免连续重复
        return signal.diff().fillna(0).clip(-1, 1)


class MACDStrategy(StrategyBase):
    """MACD：柱线由负转正买入，由正转负卖出。"""

    name = "macd"
    description = "MACD 金叉/死叉（12/26/9）"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        _, _, hist = self.macd(df["close"])
        signal = pd.Series(0, index=df.index)
        signal[hist > 0] = 1
        signal[hist < 0] = -1
        return signal.diff().fillna(0).clip(-1, 1)


class MomentumStrategy(StrategyBase):
    """动量：N 日收益率为正买入，为负卖出。"""

    name = "momentum"
    description = "N 日动量（默认 20 日）"

    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mom = df["close"].pct_change(self.window)
        signal = pd.Series(0, index=df.index)
        signal[mom > 0] = 1
        signal[mom < 0] = -1
        return signal.diff().fillna(0).clip(-1, 1)


class BreakoutStrategy(StrategyBase):
    """突破：价格创 N 日新高做多，新低做空（期货/趋势友好）。"""

    name = "breakout"
    description = "N 日通道突破（默认 20 日）"

    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        hi = df["high"].rolling(self.window).max().shift(1)
        lo = df["low"].rolling(self.window).min().shift(1)
        c = df["close"]
        signal = pd.Series(0, index=df.index)
        signal[c > hi] = 1
        signal[c < lo] = -1
        return signal.diff().fillna(0).clip(-1, 1)


class MeanReversionStrategy(StrategyBase):
    """均值回归：价格 z-score 过低买入，过高卖出。"""

    name = "mean_reversion"
    description = "价格 z-score 均值回归（默认窗口20/阈值2）"

    def __init__(self, window: int = 20, threshold: float = 2.0):
        self.window = window
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = df["close"]
        mu = c.rolling(self.window).mean()
        sd = c.rolling(self.window).std()
        z = (c - mu) / sd
        signal = pd.Series(0, index=df.index)
        signal[z < -self.threshold] = 1
        signal[z > self.threshold] = -1
        return signal.diff().fillna(0).clip(-1, 1)


REGISTRY = {
    "sma_cross": SMACrossStrategy,
    "macd": MACDStrategy,
    "momentum": MomentumStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def get_strategy(name: str, **kwargs) -> StrategyBase:
    if name not in REGISTRY:
        raise ValueError(f"未知策略: {name}，可选: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
