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
    # 隐性修复：spread 非有限(如 NaN) 原实现直接 abs(spread)/2 会泄漏 NaN 到成交价，
    # 下游净值/绩效随之被污染且无任何报错。这里统一经 _as_float 守卫后取绝对值。
    spread = abs(_as_float(spread, "spread"))
    sign = 1 if qty > 0 else -1
    slip = slippage_cost(price, qty, adv, impact_coef) + spread / 2
    return float(price + sign * slip)


def participation_rate(qty: float, adv: float) -> float:
    """参与率 |qty|/ADV（新增能力）：衡量订单对市场的相对冲击强度。"""
    qty = _as_float(qty, "qty")
    adv = _as_float(adv, "adv")
    if adv <= 0:
        return 0.0
    return float(abs(qty) / adv)


def market_impact(price: float, qty: float, adv: float, sigma: float,
                  horizon: float = 1.0, eta: float = 0.1,
                  gamma: float = 0.1) -> float:
    """Almgren-Chriss 风格市场冲击成本（新增能力，优于线性近似）。

    临时冲击(√参与率) + 永久冲击(线性) 之和，按价格缩放：

        cost = price * (eta * sigma * sqrt(part / horizon) + gamma * part)

    其中 part = |qty| / adv。用于更真实的交易成本/冲击预估。
    """
    price = _as_float(price, "price")
    qty = _as_float(qty, "qty")
    adv_v = _as_float(adv, "adv")
    sigma = _as_float(sigma, "sigma")
    if horizon <= 0:
        raise ValueError("horizon 必须为正")
    if eta < 0 or gamma < 0:
        raise ValueError("eta/gamma 冲击系数必须非负")
    if adv_v <= 0:
        return 0.0
    part = abs(qty) / adv_v
    tmp = eta * sigma * (part / horizon) ** 0.5
    perm = gamma * part
    return float(price * (tmp + perm))


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

