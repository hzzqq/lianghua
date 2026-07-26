"""K 线重采样：任意频率聚合为 OHLCV。"""
from __future__ import annotations

import pandas as pd

_BASE = ["open", "high", "low", "close"]
_VOL_ALIASES = ("volume", "vol", "Volume", "VOL", "amount", "turnover")


def _resolve_volume(df):
    if df is None:
        return None
    for c in _VOL_ALIASES:
        if c in df.columns:
            return c
    return None


def resample_ohlcv(df: pd.DataFrame, rule: str = "W",
                   label: str | None = None,
                   closed: str | None = None) -> pd.DataFrame:
    """df 需含 open/high/low/close（volume/vol 等别名自动识别）且 index 为日期。

    支持 'W'(周)/'ME'(月)/'Q'/自定义如 '2H'。volume 列名会被规范为 'volume'。
    显式守卫：缺失基础列抛清晰异常；空输入安全返回空表。
    label/closed 一并传入 resample，避免未来版本告警。
    """
    if df is None or len(df) == 0:
        cols = ["date"] + _BASE + (["volume"] if _resolve_volume(df) else [])
        return pd.DataFrame(columns=cols)

    missing = [c for c in _BASE if c not in df.columns]
    if missing:
        raise ValueError(f"resample_ohlcv 缺少必要列: {missing}（需 {_BASE}）")

    d = df.copy()
    d.index = pd.to_datetime(d.index)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    vol_col = _resolve_volume(d)
    if vol_col is not None:
        agg[vol_col] = "sum"

    if label is None and closed is None:
        res = d.resample(rule)
    else:
        _label = label or "right"
        _closed = closed or _label
        res = d.resample(rule, label=_label, closed=_closed)

    out = res.agg(agg)
    if vol_col is not None and vol_col != "volume":
        out = out.rename(columns={vol_col: "volume"})
    out = out.dropna()
    out.index = out.index.strftime("%Y-%m-%d")
    return out.reset_index().rename(columns={"index": "date"})


def resample_vwap(df: pd.DataFrame, rule: str = "W",
                  price: str = "close",
                  label: str | None = None,
                  closed: str | None = None) -> pd.DataFrame:
    """OHLCV 重采样并额外给出每桶 VWAP（成交量加权均价）列，作为可观测价格基准。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date"] + _BASE + ["volume", "vwap"])
    if price not in df.columns:
        raise ValueError(f"resample_vwap 缺少价格列: {price!r}")
    vol_col = _resolve_volume(df)
    if vol_col is None:
        raise ValueError("resample_vwap 需要成交量列(volume/vol/amount 等)以计算 VWAP")

    base = resample_ohlcv(df, rule, label, closed)
    d = df.copy()
    d.index = pd.to_datetime(d.index)
    kw = {}
    if label is not None or closed is not None:
        _label = label or "right"
        _closed = closed or _label
        kw = dict(label=_label, closed=_closed)
    prod = (d[price] * d[vol_col]).resample(rule, **kw).sum()
    vol = d[vol_col].resample(rule, **kw).sum()
    vwap = prod / vol.replace(0, pd.NA)
    out = base.set_index("date")
    out.index = pd.to_datetime(out.index)
    out["vwap"] = vwap
    out = out.dropna(subset=["vwap"])
    out.index = out.index.strftime("%Y-%m-%d")
    return out.reset_index().rename(columns={"index": "date"})
