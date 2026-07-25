"""配对交易：价差 z-score 均值回归。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pair_signal(y: pd.Series, x: pd.Series, lookback: int = 60,
                entry: float = 2.0, exit: float = 0.5) -> pd.Series:
    """对 y/x 做 OLS 对冲后价差 z-score，超卖做多价差、超买卖出。

    返回：1=做多价差(y多x空)，-1=做空价差，0=空仓。
    """
    df = pd.concat([y.astype(float), x.astype(float)], axis=1).dropna()
    df.columns = ["y", "x"]
    beta = (df["y"].cov(df["x"]) / df["x"].var())
    spread = df["y"] - beta * df["x"]
    z = (spread - spread.rolling(lookback).mean()) / (spread.rolling(lookback).std() + 1e-12)
    sig = pd.Series(0, index=z.index)
    pos = 0
    for i in range(len(z)):
        if pd.isna(z.iloc[i]):
            sig.iloc[i] = pos
            continue
        if pos == 0 and z.iloc[i] < -entry:
            pos = 1
        elif pos == 0 and z.iloc[i] > entry:
            pos = -1
        elif pos != 0 and abs(z.iloc[i]) < exit:
            pos = 0
        sig.iloc[i] = pos
    return sig.rename("pair_signal")
