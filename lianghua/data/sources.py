"""数据多源适配器（迭代 181+，超自驱动目标）。

把「从哪里取行情」抽象成统一 Source 接口，便于在 AKShare / BaoStock /
本地合成源之间切换与降级。所有源返回统一的 OHLCV DataFrame
（date, open, high, low, close, volume），取不到返回 None。

纯 pandas/numpy（akshare/baostock 为可选依赖，缺失时该源自动降级）。
"""
from __future__ import annotations

import hashlib
import time
import numpy as np
import pandas as pd

# 各源最近一次失败原因，供 UI/调用方主动提示（避免"无数据"三连却不知为何）
_LAST_ERRORS: dict[str, str] = {}


def _synth_frame(symbol: str, start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """确定性几何随机游走（合成演示行情），保证离线可用。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    # 隐性修复：start > end 会让 bdate_range 直接抛 ValueError；演示源应兜底
    if s > e:
        s, e = e, s
    h = int(hashlib.md5(str(symbol).encode()).hexdigest(), 16)
    rng = np.random.default_rng(h ^ seed)
    dates = pd.bdate_range(s, e)
    n = len(dates)
    if n < 2:
        n = 2
        dates = pd.bdate_range(s, periods=n)
    rets = rng.normal(0.0005, 0.015, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    close = np.maximum(close, 1.0)
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * (1.0 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, n)))
    vol = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })


class BaseSource:
    name: str = "base"

    def fetch(self, symbol: str, start: str, end: str, asset: str = "stock"):
        raise NotImplementedError


class SyntheticSource(BaseSource):
    """合成源：确定性随机行情，永远可用（离线兜底）。"""
    name = "synthetic"

    def fetch(self, symbol: str, start: str, end: str, asset: str = "stock"):
        return _synth_frame(symbol, start, end)


class AkshareSource(BaseSource):
    """AKShare 源：真实行情；缺失/失败返回 None。"""
    name = "akshare"

    def fetch(self, symbol: str, start: str, end: str, asset: str = "stock"):
        try:
            import akshare as ak
        except Exception:
            return None
        try:
            if asset == "fund":
                df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                         start_date=start.replace("-", ""),
                                         end_date=end.replace("-", ""))
                if df is None or df.empty:
                    return None
                df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                        "最低": "low", "收盘": "close", "成交量": "volume"})
            else:
                df = ak.stock_zh_a_hist(symbol=symbol.replace(".SH", "").replace(".SZ", ""),
                                        period="daily",
                                        start_date=start.replace("-", ""),
                                        end_date=end.replace("-", ""))
                if df is None or df.empty:
                    return None
                df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                        "最低": "low", "收盘": "close", "成交量": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "open", "high", "low", "close", "volume"]]
            return validate_ohlcv(df)
        except Exception:
            return None


class BaoStockSource(BaseSource):
    """BaoStock 源：真实行情；缺失/失败返回 None。"""
    name = "baostock"

    def fetch(self, symbol: str, start: str, end: str, asset: str = "stock"):
        try:
            import baostock as bs
        except Exception:
            return None
        try:
            bs.login()
            code = symbol
            rs = bs.query_history_k_data_plus(
                code, "date,open,high,low,close,volume",
                start_date=start, end_date=end, frequency="d", adjustflag="2")
            data = []
            while (rs.error_code == "0") and rs.next():
                data.append(rs.get_row_data())
            bs.logout()
            if not data:
                return None
            df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna()
            return validate_ohlcv(df)
        except Exception:
            return None


REGISTRY = {
    "synthetic": SyntheticSource,
    "akshare": AkshareSource,
    "baostock": BaoStockSource,
}


def validate_ohlcv(df, require_positive_close: bool = True) -> pd.DataFrame:
    """校验并清洗 OHLCV DataFrame（新增数据质量门）。

    - 必须包含 date/open/high/low/close/volume 六列，否则抛 ValueError；
    - 行情列强制转数值并剔除含 NaN 的行；
    - 默认要求 close > 0，剔除非正价（避免下游 pct_change 产生 inf）；
    - 清洗后为空同样抛 ValueError。
    真实数据源在返回前调用本函数，保证传给下游的行情是干净的。
    """
    if df is None:
        raise ValueError("OHLCV 为 None")
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in getattr(df, "columns", [])]
    if missing:
        raise ValueError(f"OHLCV 缺少列: {missing}")
    out = df.copy()
    ohlc = ["open", "high", "low", "close"]
    out[ohlc] = out[ohlc].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=ohlc)
    if require_positive_close:
        out = out[out["close"] > 0]
    if out.empty:
        raise ValueError("OHLCV 清洗后为空（无有效行情行）")
    return out.reset_index(drop=True)


def list_sources() -> list[str]:
    """可用数据源名称列表。"""
    return list(REGISTRY.keys())


def make_source(name: str) -> BaseSource:
    if name not in REGISTRY:
        raise ValueError(f"未知数据源: {name}，可选 {list_sources()}")
    return REGISTRY[name]()


def fetch_from(name: str, symbol: str, start: str, end: str, asset: str = "stock",
               retries: int = 0, backoff: float = 0.0) -> pd.DataFrame | None:
    """从指定源取数；该源不可用或失败返回 None。

    retries/backoff: 真实源（akshare/baostock）取数时按 ``retries`` 次重试，
        间隔 ``backoff`` 秒（默认不重试），扛瞬时网络抖动。每次失败原因记入
        ``_LAST_ERRORS[name]``，可由 ``last_error(name)`` 取出主动提示用户。
    """
    src = make_source(name)
    attempts = max(1, int(retries) + 1)
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            df = src.fetch(symbol, start, end, asset)
            if df is not None and not (isinstance(df, pd.DataFrame) and df.empty):
                _LAST_ERRORS.pop(name, None)
                return df
        except Exception as exc:  # noqa: BLE001 - 真实源异常需记录后降级
            last_exc = exc
        if i < attempts - 1 and backoff > 0:
            time.sleep(backoff)
    if last_exc is not None:
        _LAST_ERRORS[name] = f"{type(last_exc).__name__}: {last_exc}"
    else:
        _LAST_ERRORS[name] = "返回空（依赖缺失或无数据）"
    return None


def fetch_any(symbol: str, start: str, end: str, asset: str = "stock",
              order: list[str] | None = None,
              retries: int = 0, backoff: float = 0.0) -> pd.DataFrame | None:
    """按优先级尝试多个源，返回第一个非空 DataFrame（全失败返回 None）。

    retries/backoff 透传给每个源（见 ``fetch_from``）；单源失败原因记入
    ``_LAST_ERRORS``，便于排障。
    """
    order = order or ["akshare", "baostock", "synthetic"]
    for nm in order:
        df = fetch_from(nm, symbol, start, end, asset, retries=retries, backoff=backoff)
        if df is not None and not df.empty:
            return df
    return None


def last_error(name: str) -> str | None:
    """返回某数据源最近一次失败原因（无则 None）。"""
    return _LAST_ERRORS.get(name)


__all__ = ["BaseSource", "SyntheticSource", "AkshareSource", "BaoStockSource",
           "REGISTRY", "list_sources", "make_source", "fetch_from", "fetch_any",
           "validate_ohlcv"]
