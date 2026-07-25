"""策略基类：与回测引擎解耦的信号生成器。

任何策略只需实现 generate_signals(df) -> pd.Series，
序列取值：1=买入, -1=卖出, 0=持有。
"""
from __future__ import annotations

import pandas as pd


class StrategyBase:
    """所有策略的抽象基类。"""

    name: str = "base"
    description: str = "策略基类"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """根据行情产生交易信号。子类必须重写。"""
        raise NotImplementedError("子类需实现 generate_signals()")

    # 工具方法：常用指标
    @staticmethod
    def sma(close: pd.Series, window: int) -> pd.Series:
        return close.rolling(window).mean()

    @staticmethod
    def ema(close: pd.Series, window: int) -> pd.Series:
        return close.ewm(span=window, adjust=False).mean()

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = StrategyBase.ema(close, fast)
        ema_slow = StrategyBase.ema(close, slow)
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = (dif - dea) * 2
        return dif, dea, hist
