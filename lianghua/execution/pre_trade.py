"""盘前风控检查（迭代 181+，超自驱动目标）。

在订单提交前做功用/合规性校验：资金充足、数量为正、单笔/总持仓不超限、
非禁止标的。返回 (ok: bool, reason: str)，便于 broker / UI 拦截违规单。

非有限值（NaN/inf）守卫：这是 LiveEngine 落单前的最后一道闸门，一旦放行就是
真下单。原实现只有 price 写对了（`not (price > 0)` 能拦住 NaN），其余全是
`a > b` 形式的朴素比较——NaN 让它们恒为 False，于是 **NaN 数量的订单、NaN
现金的账户可以一路通过全部检查**。现在对每个参与比较的数字都先验有限性，
拿不准就拒单（fail-closed）。
"""
from __future__ import annotations

from ..core.numeric import is_finite_num
from .brokers.base import Order


def pre_trade_check(order: Order, account: dict,
                    limits: dict | None = None) -> tuple[bool, str]:
    """盘前风控。

    account : {"cash": float, "equity": float, "positions": {symbol: qty}, ...}
    limits  : {"max_order_value": float, "max_position_value": float,
               "min_qty": int, "blocked": [symbol...]}

    返回 (是否通过, 原因)。
    """
    limits = limits or {}
    min_qty = int(limits.get("min_qty", 0))
    if not is_finite_num(order.qty):
        return False, f"数量必须为有限数，收到 {order.qty}"
    if order.qty <= min_qty:
        return False, f"数量须为正且 > min_qty({min_qty})，收到 {order.qty}"
    side = str(order.side).upper()
    if side not in ("BUY", "SELL"):
        return False, f"side 非法：{order.side}"
    if not (is_finite_num(order.price) and float(order.price) > 0):
        return False, f"价格必须为正数，收到 {order.price}"
    if order.symbol in (limits.get("blocked") or []):
        return False, f"标的 {order.symbol} 在禁止清单"
    notional = float(order.qty) * float(order.price)
    max_ov = float(limits.get("max_order_value", float("inf")))
    if notional > max_ov:
        return False, f"单笔金额 {notional:.0f} 超过上限 {max_ov:.0f}"
    if side == "BUY":
        raw_cash = account.get("cash", 0.0)
        if raw_cash is None:
            raw_cash = 0.0
        if not is_finite_num(raw_cash):
            # `notional > NaN` 恒为 False，资金充足检查会被静默跳过。
            return False, f"账户可用现金不可用（非有限数）：{raw_cash!r}"
        cash = float(raw_cash)
        if notional > cash:
            return False, f"资金不足：需 {notional:.0f} 可用 {cash:.0f}"
    pos = account.get("positions", {})
    raw_pos = pos.get(order.symbol, 0)
    if not is_finite_num(raw_pos):
        return False, f"标的 {order.symbol} 当前持仓不可用（非有限数）：{raw_pos!r}"
    cur_qty = abs(float(raw_pos))
    max_pv = float(limits.get("max_position_value", float("inf")))
    if (cur_qty + float(order.qty)) * float(order.price) > max_pv:
        return False, f"持仓金额将超过上限 {max_pv:.0f}"
    return True, "通过"


__all__ = ["pre_trade_check"]
