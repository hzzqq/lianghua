"""盘前风控检查（迭代 181+，超自驱动目标）。

在订单提交前做功用/合规性校验：资金充足、数量为正、单笔/总持仓不超限、
非禁止标的。返回 (ok: bool, reason: str)，便于 broker / UI 拦截违规单。
"""
from __future__ import annotations

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
    if order.qty <= int(limits.get("min_qty", 0)):
        return False, f"数量须为正（min_qty={limits.get('min_qty', 0)}）"
    if order.symbol in (limits.get("blocked") or []):
        return False, f"标的 {order.symbol} 在禁止清单"
    notional = order.qty * order.price
    max_ov = float(limits.get("max_order_value", float("inf")))
    if notional > max_ov:
        return False, f"单笔金额 {notional:.0f} 超过上限 {max_ov:.0f}"
    if order.side == "BUY":
        raw_cash = account.get("cash", 0.0)
        cash = float(raw_cash) if raw_cash is not None else 0.0
        if notional > cash:
            return False, f"资金不足：需 {notional:.0f} 可用 {cash:.0f}"
    pos = account.get("positions", {})
    cur_qty = abs(pos.get(order.symbol, 0))
    max_pv = float(limits.get("max_position_value", float("inf")))
    if (cur_qty + order.qty) * order.price > max_pv:
        return False, f"持仓金额将超过上限 {max_pv:.0f}"
    return True, "通过"


__all__ = ["pre_trade_check"]
