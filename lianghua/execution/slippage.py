"""冲击成本 / 滑点模型。"""
from __future__ import annotations

import math


def _as_float(x, name: str) -> float:
    """把输入规整为有限 float；非法/缺失/NaN/inf 一律抛清晰异常。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是有限数值，收到 {x!r}")
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限数值，收到 {x!r}")
    return v


def slippage_cost(price: float, qty: float, adv: float,
                   impact_coef: float = 0.1) -> float:
    """按参与率估算冲击成本（占价格绝对值的金额）。

    显式守卫：adv/price/qty 非法或非有限会抛清晰异常；impact_coef<0（误配
    成负回扣）会被钳为 0，避免静默产出负成本；adv<=0（无流动性信息）退化为 0。
    """
    price = _as_float(price, "price")
    qty = _as_float(qty, "qty")
    adv = _as_float(adv, "adv")
    if impact_coef < 0:
        impact_coef = 0.0
    if adv <= 0:
        return 0.0
    part = abs(qty) / adv
    return float(price * impact_coef * part)


def fill_price(price: float, qty: float, adv: float,
                spread: float = 0.0, impact_coef: float = 0.1) -> float:
    """考虑买卖价差方向与冲击后的实际成交价。

    买（qty>0）支付更高价，卖（qty<0）收到更低价；价差与冲击均按方向施加。
    """
    price = _as_float(price, "price")
    qty = _as_float(qty, "qty")
    if impact_coef < 0:
        impact_coef = 0.0
    sign = 1 if qty > 0 else -1
    slip = slippage_cost(price, qty, adv, impact_coef) + abs(spread) / 2
    return float(price + sign * slip)


def slippage_fraction(price: float, qty: float, adv: float,
                      spread: float = 0.0, impact_coef: float = 0.1) -> float:
    """单边冲击+价差成本占价格的比例（可观测的交易成本指标，0~1）。"""
    price = _as_float(price, "price")
    if price == 0:
        return 0.0
    cost = abs(fill_price(price, qty, adv, spread, impact_coef) - price)
    return float(cost / abs(price))


def round_trip_cost(price: float, qty: float, adv: float,
                    spread: float = 0.0, impact_coef: float = 0.1) -> float:
    """一笔买卖往返（先买后卖）的总冲击+价差成本（金额），用于换手估算。"""
    buy = fill_price(price, abs(qty), adv, spread, impact_coef)
    sell = fill_price(price, -abs(qty), adv, spread, impact_coef)
    return float(abs(buy - sell))

