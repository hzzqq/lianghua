"""数据多源适配器（迭代 181+，超自驱动目标）。

把「从哪里取行情」抽象成统一 Source 接口，便于在 AKShare / BaoStock /
本地合成源之间切换与降级。所有源返回统一的 OHLCV DataFrame
（date, open, high, low, close, volume），取不到返回 None。

纯 pandas/numpy（akshare/baostock 为可选依赖，缺失时该源自动降级）。
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd


def _synth_frame(symbol: str, start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """确定性几何随机游走（合成演示行情），保证离线可用。"""
    h = int(hashlib.md5(str(symbol).encode()).hexdigest(), 16)
    rng = np.random.default_rng(h ^ seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    if n < 2:
        n = 2
        dates = pd.bdate_range(start, periods=n)
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
            return df[["date", "open", "high", "low", "close", "volume"]]
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
            return df.dropna()
        except Exception:
            return None


REGISTRY = {
    "synthetic": SyntheticSource,
    "akshare": AkshareSource,
    "baostock": BaoStockSource,
}


def list_sources() -> list[str]:
    """可用数据源名称列表。"""
    return list(REGISTRY.keys())


def make_source(name: str) -> BaseSource:
    if name not in REGISTRY:
        raise ValueError(f"未知数据源: {name}，可选 {list_sources()}")
    return REGISTRY[name]()


def fetch_from(name: str, symbol: str, start: str, end: str, asset: str = "stock"):
    """从指定源取数；该源不可用或失败返回 None。"""
    return make_source(name).fetch(symbol, start, end, asset)


def fetch_any(symbol: str, start: str, end: str, asset: str = "stock",
              order: list[str] | None = None) -> pd.DataFrame | None:
    """按优先级尝试多个源，返回第一个非空 DataFrame（全失败返回 None）。"""
    order = order or ["akshare", "baostock", "synthetic"]
    for nm in order:
        try:
            df = fetch_from(nm, symbol, start, end, asset)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None


__all__ = ["BaseSource", "SyntheticSource", "AkshareSource", "BaoStockSource",
           "REGISTRY", "list_sources", "make_source", "fetch_from", "fetch_any"]
