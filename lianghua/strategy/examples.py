"""示例策略库：双均线 / MACD / 动量。

均继承 StrategyBase，可直接被回测引擎加载。
"""
from __future__ import annotations

import pandas as pd

from .base import StrategyBase


class SMACrossStrategy(StrategyBase):
    """双均线交叉：短均上穿长均买入，下穿卖出。"""

    name = "sma_cross"
    cn_name = "双均线交叉"
    description = "双均线交叉（默认 5/20 日）"
    detail = (
        "短周期均线（默认 5 日）上穿长周期均线（默认 20 日）视为金叉买入，"
        "下穿视为死叉卖出，是最经典的趋势跟踪入门策略。优点是简单直观、"
        "对中长期趋势捕捉稳定；缺点是均线本身滞后，震荡市会频繁交叉产生假信号。"
        "参数：快线 5、慢线 20。"
    )

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
    cn_name = "MACD 指标"
    description = "MACD 金叉/死叉（12/26/9）"
    detail = (
        "MACD = 快慢 EMA 差（DIF）与其信号线（DEA）的差值柱状图。"
        "本策略以柱状图由负转正买入、由正转负卖出，比单纯 DIF/DEA 交叉更早反映动能变化。"
        "适合中线趋势，对拐点敏感；参数：快 12、慢 26、信号 9。"
    )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        _, _, hist = self.macd(df["close"])
        signal = pd.Series(0, index=df.index)
        signal[hist > 0] = 1
        signal[hist < 0] = -1
        return signal.diff().fillna(0).clip(-1, 1)


class MomentumStrategy(StrategyBase):
    """动量：N 日收益率为正买入，为负卖出。"""

    name = "momentum"
    cn_name = "动量策略"
    description = "N 日动量（默认 20 日）"
    detail = (
        "动量效应：过去 N 日（默认 20 日）收益率为正则顺势做多、为负则做空，"
        "假设「强者恒强」。是截面与时序上都有效的经典因子，但在趋势反转时会回撤。"
        "常与波动率目标结合控制风险。参数：回望窗口 20。"
    )

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
    cn_name = "通道突破"
    description = "N 日通道突破（默认 20 日）"
    detail = (
        "突破策略：价格创 N 日（默认 20 日）新高做多、新低做空，追随已形成的趋势方向。"
        "与海龟同源但更朴素，对期货/趋势型品种友好；缺点是突破后易回调假突破，"
        "需配合成交量或波动过滤。参数：通道窗口 20。"
    )

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
    cn_name = "均值回归"
    description = "价格 z-score 均值回归（默认窗口20/阈值2）"
    detail = (
        "均值回归：计算价格相对其 N 日滚动均值的标准差倍数（Z 分数），"
        "Z 低于 -阈值（默认 2）视为超卖买入，高于 +阈值视为超买卖出，赚取回归收益。"
        "适合区间震荡与价差类序列；强趋势中会持续偏离导致浮亏。参数：窗口 20、阈值 2。"
    )

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
