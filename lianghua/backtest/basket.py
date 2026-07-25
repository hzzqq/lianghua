"""多标的篮子组合回测（迭代24）。

给定一篮子标的 + 权重（可按资产类别自动路由到数据网关），
按归一化价格构造组合净值并对接已有 perf 指标。

basket_backtest(specs, start, end, init_cash=1_000_000)
  specs: list of dict {"symbol","weight",["asset"]}
  -> dict {equity, weights, components, metrics, attribution}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.gateway import DataGateway
from ..core.assets import AssetType, detect_asset_type
from ..perf.metrics import summary, max_drawdown, sharpe, annual_return

__all__ = ["basket_backtest", "normalize_weights"]


def normalize_weights(specs: list[dict]) -> dict:
    total = sum(abs(s["weight"]) for s in specs)
    if total == 0:
        raise ValueError("权重之和为 0")
    return {s["symbol"]: s["weight"] / total for s in specs}


def _asset_of(spec: dict) -> AssetType:
    if "asset" in spec:
        return AssetType(spec["asset"]) if not isinstance(spec["asset"], AssetType) else spec["asset"]
    return detect_asset_type(spec["symbol"])


def basket_backtest(specs: list[dict], start: str, end: str,
                    init_cash: float = 1_000_000,
                    gw: DataGateway | None = None) -> dict:
    """篮子组合回测。"""
    gw = gw or DataGateway()
    weights = normalize_weights(specs)

    # 1) 取各标的归一化价格（首日=1）
    norm = pd.DataFrame()
    closes = {}
    for spec in specs:
        sym = spec["symbol"]
        df = gw.fetch(sym, start, end, asset=_asset_of(spec))
        if df is None or df.empty:
            raise ValueError(f"无法获取标的行情: {sym}")
        price = df.set_index("date")["close"].astype(float)
        price = price / price.iloc[0]
        closes[sym] = price
        norm[sym] = price

    norm = norm.dropna()
    if norm.empty:
        raise ValueError("篮子数据为空，无法回测")

    # 2) 组合净值：各标的归一化收益按权重加权
    daily_ret = norm.pct_change().fillna(0.0)
    port_ret = daily_ret.mul(pd.Series(weights)).sum(axis=1)
    equity = (1 + port_ret).cumprod() * init_cash
    equity.iloc[0] = init_cash

    # 3) 成分贡献（各标的累计收益 × 权重，标量）
    cum = (1 + daily_ret).prod()
    contributions = {sym: float((cum[sym] - 1) * weights[sym]) for sym in norm.columns}

    metrics = summary(equity, trades=[])
    return {
        "equity": equity,
        "weights": weights,
        "components": closes,
        "metrics": metrics,
        "attribution": contributions,
        "daily_returns": port_ret,
    }
