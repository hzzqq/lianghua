"""交易前风控检查：敞口 / 流动性 / 单标的上限（迭代打磨）。

纯数据结构，不依赖具体 broker。可被任意 BaseBroker / 调度层在落单前调用。

迭代打磨（本轮回测收尾）：
- 新增 validate_order()：归一化并校验订单（symbol 非空 / side∈{BUY,SELL} /
  qty>0 / price>0 且均有限），杜绝非法订单静默进入风控。
- pre_trade_check 修复隐性 KeyError：原实现用 limits["max_position_pct"] 直接索引，
  一旦该键缺失即抛 KeyError（即便 gross==0 本不应检查）；现改为 .get 守卫。
- pre_trade_check 现为 positions 感知：根据 side 计算下单后持仓 position_after，
  并据此评估是否触及单标的敞口上限（SELL 减仓不增新敞口）。
- 新增 liquidity_check()：基于可成交量(ADV)与最大参与度(max_participation)的
  流动性闸门（机构实盘标配，防止一笔吃穿盘面）。
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["validate_order", "pre_trade_check", "liquidity_check"]

_VALID_SIDES = {"BUY", "SELL"}


def _as_finite(x, name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"order.{name} 无法转为有限浮点数: {x!r}")
    if not math.isfinite(v):
        raise ValueError(f"order.{name} 必须为有限数: {x!r}")
    return v


def validate_order(order: dict) -> dict:
    """校验并归一化订单，返回清洗后的 {symbol, side, qty, price}。

    不合法订单显式抛 ValueError，而非把脏数据交给风控后产出无意义结论。
    """
    if not isinstance(order, dict):
        raise TypeError("order 必须是 dict")
    sym = order.get("symbol")
    if not isinstance(sym, str) or not sym.strip():
        raise ValueError(f"order.symbol 必须是非空字符串: {sym!r}")
    side = str(order.get("side", "")).upper()
    if side not in _VALID_SIDES:
        raise ValueError(f"order.side 必须是 BUY/SELL，收到: {side!r}")
    qty = _as_finite(order.get("qty"), "qty")
    if qty <= 0:
        raise ValueError(f"order.qty 必须 > 0: {qty!r}")
    price = _as_finite(order.get("price"), "price")
    if price <= 0:
        raise ValueError(f"order.price 必须 > 0: {price!r}")
    return {"symbol": sym, "side": side, "qty": qty, "price": price}


def pre_trade_check(order: dict, positions: dict | None = None,
                    limits: dict | None = None) -> dict:
    """下单前风控闸门。

    order: {symbol, side, qty, price}；positions: {symbol: 当前持仓数量}（可选）；
    limits: 可选键 {gross, max_position_pct, max_gross_pct, max_order_notional}。

    返回 {ok, reasons[], notional, position_after}。
    隐性修复：原 limits["max_*"] 直接索引，键缺失即 KeyError；现用 .get 守卫且
    仅当对应上限被配置时才检查。新增 positions 感知的持仓后评估。
    """
    order = validate_order(order)
    positions = positions or {}
    limits = limits or {}
    reasons: list[str] = []
    notional = order["price"] * order["qty"]

    cur = float(positions.get(order["symbol"], 0.0) or 0.0)
    position_after = cur + order["qty"] if order["side"] == "BUY" else cur - order["qty"]

    gross = float(limits.get("gross", 0.0) or 0.0)
    mp = limits.get("max_position_pct")
    if gross > 0 and mp is not None:
        if notional > float(mp) * gross:
            reasons.append("单标的敞口超限")
    mg = limits.get("max_gross_pct")
    if gross > 0 and mg is not None:
        if notional > float(mg) * gross:
            reasons.append("总敞口超限")
    mo = limits.get("max_order_notional")
    if mo is not None and notional > float(mo):
        reasons.append("单笔名义金额超限")

    ok = len(reasons) == 0
    return {
        "ok": ok,
        "reasons": reasons,
        "notional": notional,
        "position_after": position_after,
    }


def liquidity_check(order: dict, adv: float, max_participation: float = 0.1) -> dict:
    """流动性闸门：订单成交量不得超过 adv * max_participation。

    adv: 近 N 日日均成交量（可成交量）。max_participation: 单笔允许的最大市场
    参与度（如 0.1 = 10%）。返回 {ok, reasons[], participation}。

    新增能力（此前模块只能查敞口，无法约束订单对盘面的冲击）。
    """
    order = validate_order(order)
    if not math.isfinite(adv) or adv <= 0:
        raise ValueError(f"adv(可成交量) 必须为正有限数: {adv!r}")
    if not (0.0 < max_participation < 1.0):
        raise ValueError("max_participation 必须介于 0 与 1 之间")
    participation = order["qty"] / float(adv)
    ok = participation <= max_participation
    reasons = [] if ok else ["订单量超过流动性参与度上限"]
    return {"ok": ok, "reasons": reasons, "participation": float(participation)}
