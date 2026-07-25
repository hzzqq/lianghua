"""进阶技术指标（迭代 181+，超自驱动目标）：ADX / 轴心点 / Heikin-Ashi /
Renko / DeMarker / Klinger / 蔡金波动率 / 强力指数 / KST / 之字转向 / 价格通道。

输入 OHLCV DataFrame，输出等长 Series / DataFrame。纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adx(df, period: int = 14) -> pd.Series:
    """平均趋向指数：趋势强度（不判方向），∈[0,100]，越大趋势越强。"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_h, prev_l, prev_c = high.shift(), low.shift(), close.shift()
    up = (high - prev_h)
    down = (prev_l - low)
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([(high - low).abs(), (high - prev_c).abs(), (low - prev_c).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-12)
    minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.rolling(period).mean().rename("adx")


def pivot(df) -> pd.DataFrame:
    """经典轴心点：用上一根 K 的 H/L/C 推算 PP 与 R1-S3/S1-S3 支撑阻力。"""
    h, l, c = df["high"].shift(), df["low"].shift(), df["close"].shift()
    pp = (h + l + c) / 3
    r1 = 2 * pp - l
    s1 = 2 * pp - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    r3 = h + 2 * (pp - l)
    s3 = l - 2 * (h - pp)
    return pd.DataFrame({"pp": pp, "r1": r1, "s1": s1,
                         "r2": r2, "s2": s2, "r3": r3, "s3": s3})


def heikin_ashi(df) -> pd.DataFrame:
    """Heikin-Ashi 平滑 K 线：降低噪声突出趋势。返回 ha_open/high/low/close。"""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ha_close = (o + c + h + l) / 4
    ha_open = ha_close.copy()
    for i in range(1, len(ha_close)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([ha_open, ha_close, h], axis=1).max(axis=1)
    ha_low = pd.concat([ha_open, ha_close, l], axis=1).min(axis=1)
    return pd.DataFrame({"ha_open": ha_open, "ha_high": ha_high,
                         "ha_low": ha_low, "ha_close": ha_close})


def renko(df, brick: float | None = None) -> pd.Series:
    """砖形趋势状态：价格每突破一个砖块翻转方向（1 多 / -1 空）。"""
    close = df["close"].astype(float)
    if brick is None or brick <= 0:
        brick = float((close.max() - close.min()) / 50) or float(close.std() or 1.0)
    last = float(close.iloc[0])
    direc = 0
    out = []
    for p in close.values:
        if p >= last + brick:
            direc = 1
            last = last + brick
        elif p <= last - brick:
            direc = -1
            last = last - brick
        out.append(direc)
    return pd.Series(out, index=df.index, name="renko")


def demarker(df, period: int = 14) -> pd.Series:
    """DeMark 动力指标：比较当前与前期高低，∈[0,100]，>0.7 超买 / <0.3 超卖。"""
    high, low = df["high"], df["low"]
    prev_h, prev_l = high.shift(), low.shift()
    up = (high - prev_h).clip(lower=0)
    down = (prev_l - low).clip(lower=0)
    demax = up.rolling(period).mean()
    demin = down.rolling(period).mean()
    return (demax / (demax + demin + 1e-12) * 100).rename("demarker")


def klinger(df, fast: int = 34, slow: int = 55) -> pd.Series:
    """Klinger 成交量振荡器：量能趋势（放量上行+/缩量下行-）双均线差。"""
    close, vol = df["close"], df["volume"]
    trend = np.where(close > close.shift(), 1.0, -1.0)
    vf = pd.Series(vol.values * trend, index=df.index)
    return (vf.rolling(fast).mean() - vf.rolling(slow).mean()).rename("klinger")


def chaikin_volatility(df, period: int = 10) -> pd.Series:
    """蔡金波动率：价格区间（高-低）的 EMA 变化率，骤升常预示变盘。"""
    rng = (df["high"] - df["low"])
    ema = rng.ewm(span=period).mean()
    ema_prev = ema.shift(period)
    return ((ema - ema_prev) / (ema_prev + 1e-12) * 100).rename("chaikin_vol")


def force_index(df, period: int = 13) -> pd.Series:
    """强力指数：价格变动 × 成交量，EMA 平滑，反映资金推动强度。"""
    c, v = df["close"].astype(float), df["volume"].astype(float)
    fi = (c.diff() * v)
    return fi.ewm(span=period).mean().rename("force_index")


def know_sure_thing(df, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30) -> pd.Series:
    """KST 已知善恶指标：多周期 ROC 加权的趋势综合动能。"""
    c = df["close"].astype(float)
    roc1 = c.pct_change(r1).rolling(r1).mean()
    roc2 = c.pct_change(r2).rolling(r2).mean()
    roc3 = c.pct_change(r3).rolling(r3).mean()
    roc4 = c.pct_change(r4).rolling(r4).mean()
    kst = roc1 + 2 * roc2 + 3 * roc3 + 4 * roc4
    return kst.rename("kst")


def zigzag(df, pct: float = 0.05) -> pd.Series:
    """之字转向：价格相对极值反向突破 pct 时翻转方向（1 多 / -1 空）。"""
    close = df["close"].astype(float)
    last_ext = float(close.iloc[0])
    last_dir = 0
    out = []
    for p in close.values:
        if last_dir <= 0 and p >= last_ext * (1 + pct):
            last_dir, last_ext = 1, p
        elif last_dir >= 0 and p <= last_ext * (1 - pct):
            last_dir, last_ext = -1, p
        out.append(last_dir)
    return pd.Series(out, index=df.index, name="zigzag")


def price_channels(df, period: int = 20) -> pd.DataFrame:
    """价格通道：N 日最高价（上轨）与最低价（下轨）构成的轨道。"""
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    return pd.DataFrame({"upper": upper, "lower": lower})


__all__ = ["adx", "pivot", "heikin_ashi", "renko", "demarker", "klinger",
           "chaikin_volatility", "force_index", "know_sure_thing",
           "zigzag", "price_channels"]
