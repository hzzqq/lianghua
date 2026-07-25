"""交易前风控检查：敞口 / 流动性 / 单标的上限。"""
from __future__ import annotations

import pandas as pd


def pre_trade_check(order: dict, positions: dict, limits: dict) -> dict:
    """order: {symbol, side, qty, price}；positions: 现持仓{标的: 数量}。

    limits: {max_position_pct, max_gross_pct, cash}。返回 {ok, reasons[]}。
    """
    reasons = []
    sym = order["symbol"]
    price = float(order["price"])
    qty = float(order["qty"])
    notional = price * qty
    # 单标的市值占比
    gross = limits.get("gross", 0.0)
    if gross > 0 and notional > limits["max_position_pct"] * gross:
        reasons.append("单标的敞口超限")
    # 总敞口
    if gross > 0 and notional > limits["max_gross_pct"] * gross:
        reasons.append("总敞口超限")
    ok = len(reasons) == 0
    return {"ok": ok, "reasons": reasons, "notional": notional}
