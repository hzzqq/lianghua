"""进阶技术指标（迭代 101-110）：
supertrend / aroon / vortex / trix / williams_r / cmf / mfi / stoch_rsi / dpo / ppo。

全部基于 pandas/numpy，零额外依赖。输入含 OHLCV 的 DataFrame 或 close Series，
输出与输入等长的 Series（或多列 DataFrame）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tech import atr, rsi, macd


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Supertrend：基于 ATR 的趋势跟踪线。

    返回 DataFrame(st=趋势线, dir=方向 +1 多 / -1 空)。
    """
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    a = atr(df, period)
    hl2 = (h + l) / 2.0
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    n = len(c)
    st = pd.Series(index=c.index, dtype=float)
    direction = pd.Series(1, index=c.index, dtype=int)
    fu = upper.copy()
    fl = lower.copy()
    for i in range(1, n):
        fu.iloc[i] = (min(upper.iloc[i], fu.iloc[i - 1])
                      if c.iloc[i - 1] <= fu.iloc[i - 1] else upper.iloc[i])
        fl.iloc[i] = (max(lower.iloc[i], fl.iloc[i - 1])
                      if c.iloc[i - 1] >= fl.iloc[i - 1] else lower.iloc[i])
        if c.iloc[i] > fu.iloc[i - 1]:
            direction.iloc[i] = 1
        elif c.iloc[i] < fl.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
        st.iloc[i] = fl.iloc[i] if direction.iloc[i] == 1 else fu.iloc[i]
    return pd.DataFrame({"st": st, "dir": direction})


def aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    """Aroon 指标：衡量距最高/最低点的时间距离。返回 up/down (0-100)。"""
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    def _since_high(x):
        return (len(x) - 1 - int(np.argmax(x))) 

    def _since_low(x):
        return (len(x) - 1 - int(np.argmin(x)))

    up = h.rolling(period + 1).apply(lambda x: 100 * (period - _since_high(x)) / period, raw=True)
    dn = l.rolling(period + 1).apply(lambda x: 100 * (period - _since_low(x)) / period, raw=True)
    return pd.DataFrame({"up": up, "down": dn})


def vortex(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Vortex 指标：VI+ / VI- 衡量趋势方向强度。"""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    vmp = (h - l.shift(1)).abs()
    vmm = (l - h.shift(1)).abs()
    tr_sum = tr.rolling(period).sum()
    vip = vmp.rolling(period).sum() / (tr_sum + 1e-12)
    vim = vmm.rolling(period).sum() / (tr_sum + 1e-12)
    return pd.DataFrame({"vi_plus": vip, "vi_minus": vim})


def trix(series: pd.Series, period: int = 15, signal: int = 9) -> pd.DataFrame:
    """TRIX：三重指数平滑变化率。返回 trix 与 signal 线（百分比）。"""
    c = series.astype(float)
    e1 = c.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    tx = e3.pct_change() * 100
    sig = tx.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"trix": tx, "signal": sig})


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R：范围 -100..0，越接近 0 越超买。"""
    h = df["high"].astype(float).rolling(period).max()
    l = df["low"].astype(float).rolling(period).min()
    c = df["close"].astype(float)
    return -100 * (h - c) / (h - l + 1e-12)


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow：量价资金流强度，范围约 -1..1。"""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float) if "volume" in df else c * 0 + 1.0
    mfm = ((c - l) - (h - c)) / (h - l + 1e-12)
    mfv = mfm * v
    return mfv.rolling(period).sum() / (v.rolling(period).sum() + 1e-12)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index：量价加权 RSI，范围 0..100。"""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float) if "volume" in df else c * 0 + 1.0
    tp = (h + l + c) / 3.0
    mf = tp * v
    delta = tp.diff()
    pos = mf.where(delta > 0, 0.0).rolling(period).sum()
    neg = mf.where(delta < 0, 0.0).rolling(period).sum()
    mr = pos / (neg + 1e-12)
    return 100 - 100 / (1 + mr)


def stoch_rsi(series: pd.Series, period: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    """Stochastic RSI：对 RSI 再做随机指标，范围 0..1。返回 k/d。"""
    r = rsi(series, period)
    lo = r.rolling(period).min()
    hi = r.rolling(period).max()
    sr = (r - lo) / (hi - lo + 1e-12)
    k_line = sr.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return pd.DataFrame({"k": k_line, "d": d_line})


def dpo(series: pd.Series, period: int = 20) -> pd.Series:
    """Detrended Price Oscillator：去趋势价格振荡，识别周期。"""
    c = series.astype(float)
    ma = c.rolling(period).mean()
    shift = period // 2 + 1
    return c.shift(shift) - ma


def ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Percentage Price Oscillator：MACD 的百分比版本（跨标的可比）。"""
    c = series.astype(float)
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    line = (ef - es) / (es + 1e-12) * 100
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({"ppo": line, "signal": sig, "hist": hist})
