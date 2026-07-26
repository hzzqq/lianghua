"""多标的篮子组合回测（迭代24，本轮收尾打磨）。

给定一篮子标的 + 权重（可按资产类别自动路由到数据网关），
按归一化价格构造组合净值并对接已有 perf 指标。

basket_backtest(specs, start, end, init_cash=1_000_000)
  specs: list of dict {"symbol","weight",["asset"]}
  -> dict {equity, weights, components, metrics, attribution, daily_returns}

迭代打磨（本轮回测收尾）：
- 修复隐性索引错位：原实现在 norm.dropna() 之前就计算 daily_ret，导致
  equity/port_ret 仍保留被 dropna 丢弃的日期行，与 norm 索引错位、指标失真。
  现改为先 dropna 再算收益。
- 新增 validate_specs()：校验 specs 非空、每项含 symbol(str)/weight(有限数)。
- 新增 basket_daily_returns()：独立返回组合日收益序列（可复用/可观测）。
- 守卫首日出价为零/非有限（归一化除零 → inf）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.gateway import DataGateway
from ..core.assets import AssetType, detect_asset_type
from ..perf.metrics import summary

__all__ = ["basket_backtest", "normalize_weights", "validate_specs", "basket_daily_returns"]


def validate_specs(specs: list[dict]) -> None:
    """校验篮子规格：非空列表；每项含非空 symbol(str) 与有限 weight。"""
    if not isinstance(specs, (list, tuple)):
        raise TypeError("specs 必须是 list[dict]")
    if len(specs) == 0:
        raise ValueError("specs 不能为空")
    for i, s in enumerate(specs):
        if not isinstance(s, dict):
            raise TypeError(f"specs[{i}] 必须是 dict")
        sym = s.get("symbol")
        if not isinstance(sym, str) or not sym.strip():
            raise ValueError(f"specs[{i}].symbol 必须是非空字符串")
        w = s.get("weight")
        try:
            wf = float(w)
        except (TypeError, ValueError):
            raise ValueError(f"specs[{i}].weight 必须是有限数，收到 {w!r}")
        if not np.isfinite(wf):
            raise ValueError(f"specs[{i}].weight 必须为有限数，收到 {w!r}")


def normalize_weights(specs: list[dict]) -> dict:
    """权重归一化（长仓口径：取绝对值后归一，权重和为 1）。"""
    validate_specs(specs)
    total = sum(abs(float(s["weight"])) for s in specs)
    if total == 0:
        raise ValueError("权重之和为 0，无法归一化")
    return {s["symbol"]: float(s["weight"]) / total for s in specs}


def _asset_of(spec: dict) -> AssetType:
    if "asset" in spec:
        return AssetType(spec["asset"]) if not isinstance(spec["asset"], AssetType) else spec["asset"]
    return detect_asset_type(spec["symbol"])


def _normalized_frame(specs, start, end, gw) -> tuple:
    """取各标的归一化价格(首日=1)，dropna 对齐后返回 (weights, norm_df, daily_ret, closes)。"""
    weights = normalize_weights(specs)
    closes: dict = {}
    frames: dict = {}
    for spec in specs:
        sym = spec["symbol"]
        df = gw.fetch(sym, start, end, asset=_asset_of(spec))
        if df is None or df.empty:
            raise ValueError(f"无法获取标的行情: {sym}")
        price = df.set_index("date")["close"].astype(float)
        if not np.isfinite(price.iloc[0]) or price.iloc[0] == 0:
            raise ValueError(f"标的 {sym} 首日出价非法(0 或非有限)，无法归一化")
        price = price / price.iloc[0]
        closes[sym] = price
        frames[sym] = price
    norm = pd.DataFrame(frames).dropna()
    if norm.empty:
        raise ValueError("篮子无共同交易日，数据为空，无法回测")
    daily_ret = norm.pct_change().fillna(0.0)
    return weights, norm, daily_ret, closes


def basket_daily_returns(specs: list[dict], start: str, end: str,
                         gw: DataGateway | None = None) -> pd.Series:
    """新增能力：直接返回组合日收益序列（不构造净值），便于下游复用/对齐。"""
    gw = gw or DataGateway()
    _, _, daily_ret, _ = _normalized_frame(specs, start, end, gw)
    weights = normalize_weights(specs)
    return daily_ret.mul(pd.Series(weights)).sum(axis=1)


def basket_backtest(specs: list[dict], start: str, end: str,
                    init_cash: float = 1_000_000,
                    gw: DataGateway | None = None) -> dict:
    """篮子组合回测。"""
    gw = gw or DataGateway()
    validate_specs(specs)
    weights, norm, daily_ret, closes = _normalized_frame(specs, start, end, gw)
    port_ret = daily_ret.mul(pd.Series(weights)).sum(axis=1)
    equity = (1 + port_ret).cumprod() * init_cash
    equity.iloc[0] = init_cash

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
