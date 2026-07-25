"""止损管理：硬止损 / 跟踪止损 / ATR 吊灯止损。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """真实波幅 TR = max(H-L, |H-Cprev|, |L-Cprev|)。"""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rename("tr")


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = 14) -> pd.Series:
    """ATR：TR 的滚动均值（Wilder 风格的指数平滑）。"""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean().rename("atr")


def hard_stop(entry: float, stop_pct: float) -> float:
    """固定比例硬止损价：entry*(1-stop_pct)。"""
    if stop_pct <= 0:
        raise ValueError("stop_pct 必须为正")
    return entry * (1.0 - stop_pct)


def trailing_stop(price: pd.Series, entry: float, init_stop: float,
                   trail_pct: float = 0.1) -> pd.Series:
    """价格从最高点回撤超 trail_pct 触发止损，返回止损价序列。"""
    highest = price.cummax()
    stop = highest * (1 - trail_pct)
    stop = stop.where(stop >= entry * (1 - init_stop), entry * (1 - init_stop))
    return stop.rename("trailing_stop")


def atr_trailing_stop(price: pd.Series, atr: pd.Series, mult: float = 3.0) -> pd.Series:
    """ATR 吊灯止损(Chandelier Exit)：运行最高点 - mult×ATR，随价格下行移动。

    相比固定比例跟踪止损，ATR 版随标的波动自适应，高波动时让出更大空间，
    低波动时收紧，减少被噪音扫损。新增能力。
    """
    highest = price.cummax()
    return (highest - mult * atr).rename("atr_trailing_stop")


def stop_triggered(price: pd.Series, stop: pd.Series) -> pd.Series:
    """返回每根 bar 是否触发止损（价格<=止损）。"""
    return (price <= stop).rename("triggered")
