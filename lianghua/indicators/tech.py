"""常用技术指标：RSI / MACD / KDJ / BOLL / ATR / CCI / OBV / VWAP / Stochastic / Williams %R。

全部基于 pandas/numpy，零额外依赖。输入为含 OHLCV 的 DataFrame（或价格 Series），
输出为与输入等长的 Series（或 DataFrame）。

所有函数带输入守卫：缺失列、非法周期、空/非数值输入都会给出清晰错误，而非难以定位的
KeyError / TypeError / 静默 NaN。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check_period(period: int, name: str = "period") -> int:
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError(f"{name} 必须为正整数，收到 {period!r}")
    return int(period)


def _require_cols(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c not in getattr(df, "columns", [])]
    if missing:
        raise ValueError(f"指标需要列 {missing}，但输入仅有 {list(df.columns)}")


def _as_float(series, name: str = "series") -> pd.Series:
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception:
            raise TypeError(f"{name} 必须是 1-D array-like，收到 {type(series).__name__}")
    s = series.astype(float)
    if s.empty:
        raise ValueError(f"{name} 不能为空")
    return s


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    period = _check_period(period, "period")
    s = _as_float(series, "series")
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    avg_up = up.rolling(period).mean()
    avg_dn = dn.rolling(period).mean()
    rs = avg_up / (avg_dn + 1e-12)
    out = 100 - 100 / (1.0 + rs)
    # 平坦行情（无涨跌）时 RS 退化为 0 会错误给出 RSI=0，应中性化为 50。
    out = out.where((avg_up + avg_dn) > 1e-9, 50.0)
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast = _check_period(fast, "fast")
    slow = _check_period(slow, "slow")
    signal = _check_period(signal, "signal")
    s = _as_float(series, "series")
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


def kdj(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    n = _check_period(n, "n")
    k_smooth = _check_period(k_smooth, "k_smooth")
    d_smooth = _check_period(d_smooth, "d_smooth")
    _require_cols(df, ["high", "low", "close"])
    h = df["high"].astype(float).rolling(n).max()
    l = df["low"].astype(float).rolling(n).min()
    rsv = (df["close"].astype(float) - l) / (h - l + 1e-12) * 100
    k = rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"k": k, "d": d, "j": j})


def boll(series: pd.Series, period: int = 20, nb: float = 2.0) -> pd.DataFrame:
    period = _check_period(period, "period")
    s = _as_float(series, "series")
    m = s.rolling(period).mean()
    sd = s.rolling(period).std()
    return pd.DataFrame({"mid": m, "upper": m + nb * sd, "lower": m - nb * sd})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    period = _check_period(period, "period")
    _require_cols(df, ["high", "low", "close"])
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev).abs(), (l - prev).abs()))
    return tr.rolling(period).mean()


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    period = _check_period(period, "period")
    _require_cols(df, ["high", "low", "close"])
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    tp = (h + l + c) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md + 1e-12)


def obv(df: pd.DataFrame) -> pd.Series:
    _require_cols(df, ["close", "volume"])
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    direction = np.sign(c.diff().fillna(0))
    return (direction * v).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    _require_cols(df, ["close", "volume"])
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    cum_v = v.cumsum()
    # 累计成交量为 0（如首行 volume=0）时避免 0/0 产生 NaN/inf。
    out = (c * v).cumsum() / cum_v.replace(0, np.nan)
    return out


def stochastic(df: pd.DataFrame, n: int = 14, d_period: int = 3, smooth: int = 3) -> pd.DataFrame:
    """随机指标：返回 %K（慢速）与 %D。两者均落在 [0, 100]。"""
    n = _check_period(n, "n")
    d_period = _check_period(d_period, "d_period")
    smooth = _check_period(smooth, "smooth")
    _require_cols(df, ["high", "low", "close"])
    h = df["high"].astype(float).rolling(n).max()
    l = df["low"].astype(float).rolling(n).min()
    rsv = (df["close"].astype(float) - l) / (h - l + 1e-12) * 100
    k = rsv.rolling(d_period).mean()
    d = k.rolling(smooth).mean()
    return pd.DataFrame({"k": k, "d": d})


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """威廉指标 %R，取值范围 [-100, 0]。"""
    period = _check_period(period, "period")
    _require_cols(df, ["high", "low", "close"])
    h = df["high"].astype(float).rolling(period).max()
    l = df["low"].astype(float).rolling(period).min()
    return (h - df["close"].astype(float)) / (h - l + 1e-12) * -100


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """黄金交叉：a 上穿 b（前一根 a<=b 且当前 a>b），返回布尔 Series（首值为 False）。

    信号原语，供均线/指标交叉策略复用；对齐索引后比较，避免错位误判。
    """
    a = _as_float(a, "a")
    b = _as_float(b, "b")
    a, b = a.align(b, join="outer", fill_value=np.nan)
    prev = (a < b) | (a <= b)  # 上穿前 a 应在 b 下方
    cur = a > b
    return (prev.shift(1).fillna(False)) & cur


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """死亡交叉：a 下穿 b（前一根 a>=b 且当前 a<b），返回布尔 Series（首值为 False）。"""
    a = _as_float(a, "a")
    b = _as_float(b, "b")
    a, b = a.align(b, join="outer", fill_value=np.nan)
    prev = (a > b) | (a >= b)
    cur = a < b
    return (prev.shift(1).fillna(False)) & cur
