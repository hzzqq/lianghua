"""execution.orders 测试：括号单方向修复、OCO、追踪止损。"""
import pytest

from lianghua.execution.orders import (
    bracket_order, oco_order, evaluate_bracket, evaluate_oco,
    trailing_stop_order, evaluate_trailing_stop,
)
from lianghua.core.assets import AssetType


def test_bracket_buy_valid_and_eval():
    o = bracket_order("A", "BUY", 100, 100.0, 95.0, 108.0, asset_type=AssetType.STOCK)
    # 触达入场价
    assert evaluate_bracket(o, 101.0) == ["entry"]
    # 跌破止损
    assert evaluate_bracket(o, 94.0) == ["stop"]
    # 涨破止盈（无状态评估：入场条件仍满足，故 entry/target 同时返回）
    assert evaluate_bracket(o, 109.0) == ["entry", "target"]


def test_bracket_sell_direction_fixed():
    # SELL 空单：止损在上方、止盈在下方
    o = bracket_order("A", "SELL", 100, 100.0, 105.0, 95.0, asset_type=AssetType.STOCK)
    # 价格跌破入场 -> 入场
    assert evaluate_bracket(o, 99.0) == ["entry"]
    # 价格涨破止损 -> 触发止损（修复前的反向 bug 会漏判）
    assert evaluate_bracket(o, 107.0) == ["stop"]
    # 价格跌破止盈（无状态评估：entry 条件仍满足）
    assert evaluate_bracket(o, 93.0) == ["entry", "target"]


def test_bracket_invalid_prices_raise():
    with pytest.raises(ValueError):
        bracket_order("A", "BUY", 100, 100.0, 102.0, 108.0)  # 止损高于入场
    with pytest.raises(ValueError):
        bracket_order("A", "SELL", 100, 100.0, 95.0, 105.0)  # SELL 止损低于入场


def test_oco_cancel_is_marker_not_order():
    o = oco_order("A", "BUY", 100, 90.0, 110.0)
    assert isinstance(o["cancel"], dict)
    assert o["cancel"]["cancel"] is True
    assert "price" not in o["cancel"]  # 不再是魔法价 0.0 的 Order
    assert evaluate_oco(o, 89.0) == ["a"]
    assert evaluate_oco(o, 111.0) == ["b"]


def test_trailing_stop_buy():
    st = trailing_stop_order("A", "BUY", 100, 100.0, 0.05)
    assert st["activated"] is False
    # 未到激活价：不触发
    assert evaluate_trailing_stop(st, 99.0) is None
    # 到达激活价，止损初始化为 95；继续上涨不触发且止损上移
    assert evaluate_trailing_stop(st, 110.0) is None
    assert st["stop_price"] == pytest.approx(104.5)
    # 回落至止损价下方：触发并返回 Order
    order = evaluate_trailing_stop(st, 100.0)
    assert order is not None
    assert order.price == pytest.approx(104.5)
    assert order.side == "BUY"


def test_trailing_stop_sell():
    st = trailing_stop_order("A", "SELL", 100, 100.0, 0.05)
    assert evaluate_trailing_stop(st, 101.0) is None  # 未跌破激活
    assert evaluate_trailing_stop(st, 90.0) is None   # 激活, 止损=94.5
    assert st["stop_price"] == pytest.approx(94.5)
    order = evaluate_trailing_stop(st, 96.0)           # 涨破止损 -> 触发
    assert order is not None
    assert order.side == "SELL"
    assert order.price == pytest.approx(94.5)


def test_trailing_stop_bad_pct():
    with pytest.raises(ValueError):
        trailing_stop_order("A", "BUY", 100, 100.0, 1.5)
    with pytest.raises(ValueError):
        trailing_stop_order("A", "BUY", 100, -10.0, 0.05)
