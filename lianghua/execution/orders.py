"""高级订单类型：括号单(Bracket) / 二选一单(OCO) / 追踪止损(Trailing Stop)。

纯数据结构 + 触发评估，不依赖具体 broker；返回的 Order 可直接交给任意
BaseBroker.submit。支持股票/期货/期权（asset_type 透传）。

迭代强化：
- bracket_order 增加价位合法性校验（止损/止盈必须位于入场价正确一侧）。
- evaluate_bracket 纠正 SELL(空) 场景下止损/止盈触发方向的隐性反向 bug。
- oco_order 的撤单腿不再用魔法价 0.0，改为显式 cancel 标记 dict。
- 新增 trailing_stop_order / evaluate_trailing_stop（追踪止损，机构常用）。
"""
from __future__ import annotations

from .brokers.base import Order
from ..core.assets import AssetType


def bracket_order(symbol: str, side: str, qty: int, entry: float,
                  stop_loss: float, take_profit: float,
                  asset_type: AssetType = AssetType.STOCK) -> dict:
    """括号单：入场单 + 止损单 + 止盈单三件套。

    返回 {"entry": Order, "stop": Order, "target": Order, "kind": "bracket"}。
    入场 side=BUY 时，止损在下方、止盈在上方；side=SELL 时相反。

    隐性修复：校验 stop_loss / take_profit 必须位于 entry 的正确一侧，
    否则入场瞬间即触发止损（无效括号单），直接抛错。
    """
    side = side.upper()
    opp = "SELL" if side == "BUY" else "BUY"
    if side == "BUY":
        if not (stop_loss < entry < take_profit):
            raise ValueError(
                f"BUY 括号单需 stop_loss({stop_loss}) < entry({entry}) < take_profit({take_profit})")
    else:
        if not (stop_loss > entry > take_profit):
            raise ValueError(
                f"SELL 括号单需 stop_loss({stop_loss}) > entry({entry}) > take_profit({take_profit})")
    return {
        "kind": "bracket",
        "entry": Order(symbol, side, qty, entry, asset_type),
        "stop": Order(symbol, opp, qty, stop_loss, asset_type),
        "target": Order(symbol, opp, qty, take_profit, asset_type),
    }


def oco_order(symbol: str, side: str, qty: int, price_a: float, price_b: float,
              asset_type: AssetType = AssetType.STOCK) -> dict:
    """二选一单(OCO)：两笔同向单，触发其一即撤销另一（撤销由引擎处理）。

    返回 {"a": Order, "b": Order, "cancel": {...}, "kind": "oco"}。
    隐性修复：撤单腿用显式 cancel 标记，不再用 price=0.0 的 Order（魔法值易误提交）。
    """
    side = side.upper()
    opp = "SELL" if side == "BUY" else "BUY"
    return {
        "kind": "oco",
        "a": Order(symbol, side, qty, price_a, asset_type),
        "b": Order(symbol, side, qty, price_b, asset_type),
        "cancel": {"cancel": True, "symbol": symbol, "side": opp, "qty": qty},
    }


def evaluate_bracket(bkt: dict, market_price: float) -> list[str]:
    """评估括号单触发：返回应提交的订单键列表（entry/stop/target）。

    隐性修复：止损/止盈触发方向需随入场方向翻转——
    BUY: 止损在下方(price<=stop)，止盈在上方(price>=target)；
    SELL: 止损在上方(price>=stop)，止盈在下方(price<=target)。
    """
    fired: list[str] = []
    entry = bkt.get("entry")
    if entry is None:
        return fired
    side = entry.side
    if (side == "BUY" and market_price >= entry.price) or \
       (side == "SELL" and market_price <= entry.price):
        fired.append("entry")
    stop = bkt.get("stop")
    if stop is not None:
        if (side == "BUY" and market_price <= stop.price) or \
           (side == "SELL" and market_price >= stop.price):
            fired.append("stop")
    target = bkt.get("target")
    if target is not None:
        if (side == "BUY" and market_price >= target.price) or \
           (side == "SELL" and market_price <= target.price):
            fired.append("target")
    return fired


def evaluate_oco(oco: dict, market_price: float) -> list[str]:
    """评估 OCO 触发：价格先到哪一档即触发该腿。"""
    if "a" in oco and market_price <= oco["a"].price:
        return ["a"]
    if "b" in oco and market_price >= oco["b"].price:
        return ["b"]
    return []


def trailing_stop_order(symbol: str, side: str, qty: int, activation_price: float,
                        trailing_pct: float,
                        asset_type: AssetType = AssetType.STOCK) -> dict:
    """追踪止损单（新增能力）。

    - activation_price: 触发激活的价位（BUY 需价格涨破、SELL 需跌破）。
    - trailing_pct: 回撤比例（如 0.05 = 5%），止损价随有利方向动态上/下移。

    返回可变 state dict，交由 evaluate_trailing_stop 逐 tick 评估并更新 stop_price。
    """
    side = side.upper()
    if not (0.0 < trailing_pct < 1.0):
        raise ValueError("trailing_pct 必须介于 0 与 1 之间（如 0.05 表示 5%）")
    if activation_price <= 0:
        raise ValueError("activation_price 必须为正")
    init_stop = (activation_price * (1 - trailing_pct) if side == "BUY"
                 else activation_price * (1 + trailing_pct))
    return {
        "kind": "trailing_stop",
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "asset_type": asset_type,
        "activation_price": activation_price,
        "trailing_pct": trailing_pct,
        "stop_price": init_stop,
        "activated": False,
    }


def evaluate_trailing_stop(state: dict, market_price: float):
    """评估追踪止损：原地更新 state 的 stop_price，触发则返回 Order，否则 None。

    激活后：BUY 止损随价格上涨而上移、价格跌破止损价触发；
           SELL 止损随价格下跌而下移、价格涨破止损价触发。
    """
    side = state["side"]
    if not state.get("activated"):
        if (side == "BUY" and market_price >= state["activation_price"]) or \
           (side == "SELL" and market_price <= state["activation_price"]):
            state["activated"] = True
        else:
            return None
    tp = state["trailing_pct"]
    if side == "BUY":
        new_stop = market_price * (1 - tp)
        if new_stop > state["stop_price"]:
            state["stop_price"] = new_stop
        triggered = market_price <= state["stop_price"]
    else:
        new_stop = market_price * (1 + tp)
        if new_stop < state["stop_price"]:
            state["stop_price"] = new_stop
        triggered = market_price >= state["stop_price"]
    if triggered:
        return Order(state["symbol"], side, state["qty"],
                     state["stop_price"], state["asset_type"])
    return None


__all__ = ["bracket_order", "oco_order", "evaluate_bracket", "evaluate_oco",
           "trailing_stop_order", "evaluate_trailing_stop"]
