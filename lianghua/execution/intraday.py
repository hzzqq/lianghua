"""盘中分钟级信号与监控（第十三轮·深化方向一：盘中分钟级信号）。

- `minute_signal(symbol, strategy, gateway, freq)`：取当日分钟线生成信号。
- `watch(...)`：轻量持续监控回调（不依赖 broker，仅看信号）。
- `IntradayEngine`：分钟级实盘引擎（默认 5min，决策价用实时快照）。
"""
from __future__ import annotations

import datetime as _dt
import time

import numpy as np

from ..core.assets import AssetType, detect_asset_type
from .live import LiveEngine, _resolve_strategy, _asset_val


def minute_signal(symbol, strategy, gateway, freq: str = "5min",
                  lookback: int = 240, asset=None):
    """取当日分钟线，策略生成信号。返回 (signal, last_price, df)。"""
    strat = _resolve_strategy(strategy)
    at = asset or detect_asset_type(symbol)
    today = _dt.date.today().strftime("%Y-%m-%d")
    df = gateway.fetch_minute(symbol, today, asset=_asset_val(at),
                              freq=freq, bars=lookback)
    sig = strat.generate_signals(df)
    last = int(np.sign(float(sig.iloc[-1]))) if len(sig) else 0
    last_price = float(df["close"].iloc[-1]) if len(df) else 0.0
    return last, last_price, df


def watch(symbols, strategy, gateway, freq: str = "5min", lookback: int = 240,
          interval: float = 60.0, on_signal=None, max_cycles=None):
    """轻量持续监控：仅看分钟信号，不依赖 broker，触发 on_signal(symbol, signal, price)。"""
    strat = _resolve_strategy(strategy)
    cycles = 0
    while True:
        for sym in symbols:
            try:
                sg, px, _ = minute_signal(sym, strat, gateway, freq=freq,
                                          lookback=lookback)
                if on_signal:
                    on_signal(sym, sg, px)
            except Exception as e:
                if on_signal:
                    on_signal(sym, 0, 0.0, error=str(e))
        cycles += 1
        if max_cycles is not None and cycles >= int(max_cycles):
            break
        if interval > 0:
            time.sleep(interval)


class IntradayEngine(LiveEngine):
    """分钟级实盘引擎（盘中）。默认 freq=5min，决策价用实时快照 live_quote。"""

    def __init__(self, *args, freq: str = "5min", **kwargs):
        kwargs.pop("freq", None)
        super().__init__(*args, **kwargs, freq=freq)


__all__ = ["minute_signal", "watch", "IntradayEngine"]
