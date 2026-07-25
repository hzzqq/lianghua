"""订单成交模拟：市价 / 限价撮合。"""
from __future__ import annotations

import pandas as pd


def fill_market(order: dict, bar: dict) -> float:
    """市价单按 bar 开盘（或收盘）成交。返回成交价。"""
    return float(bar.get("open", bar.get("close")))


def fill_limit(order: dict, bar: dict) -> float | None:
    """限价单：买价>=最低价才成交于限价，卖价<=最高价成交于限价。"""
    px = float(order["price"])
    if order["side"] == "BUY":
        return px if float(bar["low"]) <= px else None
    else:
        return px if float(bar["high"]) >= px else None


def simulate_orders(orders: list, bars: pd.DataFrame) -> list:
    """orders: [{side,price,type,date}]；对每单在对应 date 的 bar 上撮合。"""
    out = []
    for o in orders:
        if o["date"] not in set(bars["date"]):
            out.append({**o, "filled": None})
            continue
        bar = bars[bars["date"] == o["date"]].iloc[0].to_dict()
        px = fill_market(o, bar) if o.get("type", "MARKET") == "MARKET" else fill_limit(o, bar)
        out.append({**o, "filled": px})
    return out
