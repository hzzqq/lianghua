"""Renko 砖形趋势策略（迭代 143）。

以固定砖块(brick)尺寸把价格运动离散为上升/下降砖，跟踪当前砖方向作为趋势信号。
brick 缺省取收盘价的波动尺度（std×0.5）；输入 OHLCV DataFrame。
纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def renko_trend_signal(df: pd.DataFrame, brick: float | None = None) -> pd.Series:
    """Renko 砖形趋势：把价格运动离散为升/降砖，以当前砖方向作为趋势信号。

    brick（砖块尺寸）的取值是前视偏差的关键：
    - 给定 brick：固定尺寸，无前视（推荐）。
    - 缺省：改用**截至当前的历史标准差**（expanding），而非全样本 std。
      原实现用 ``close.std()`` 需要未来价格才能定尺寸，回测会系统性虚高。
    """
    close = df["close"].astype(float).reset_index(drop=True)
    if brick is not None:
        sizes = pd.Series(float(brick), index=close.index)
    else:
        sizes = (close.expanding().std() * 0.5).clip(lower=1e-6)

    cur = float(close.iloc[0])
    direction = 0
    bricks = []
    for i, c in enumerate(close.values.astype(float)):
        b = float(sizes.iloc[i])
        if not np.isfinite(b) or b <= 0:
            b = 1e-6
        while c >= cur + b:
            cur += b
            direction = 1
        while c <= cur - b:
            cur -= b
            direction = -1
        bricks.append(direction)

    return pd.Series(bricks, index=df.index, name="renko_trend")
