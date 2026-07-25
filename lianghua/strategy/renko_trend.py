"""Renko 砖形趋势策略（迭代 143）。

以固定砖块(brick)尺寸把价格运动离散为上升/下降砖，跟踪当前砖方向作为趋势信号。
brick 缺省取收盘价的波动尺度（std×0.5）；输入 OHLCV DataFrame。
纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def renko_trend_signal(df: pd.DataFrame, brick: float | None = None) -> pd.Series:
    close = df["close"].astype(float).reset_index(drop=True)
    if brick is None:
        brick = max(close.std() * 0.5, 1e-6)

    cur = float(close.iloc[0])
    direction = 0
    bricks = []
    for c in close.values.astype(float):
        while c >= cur + brick:
            cur += brick
            direction = 1
        while c <= cur - brick:
            cur -= brick
            direction = -1
        bricks.append(direction)

    return pd.Series(bricks, index=df.index, name="renko_trend")
