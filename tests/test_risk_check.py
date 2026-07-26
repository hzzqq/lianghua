"""迭代 45：execution/risk_check 打磨 —— 订单校验 + positions 感知敞口 + 流动性闸门。"""
import math

import pytest

from lianghua.execution.risk_check import (
    validate_order, pre_trade_check, liquidity_check,
)


def _order(**kw):
    base = {"symbol": "AAPL", "side": "BUY", "qty": 100, "price": 10.0}
    base.update(kw)
    return base


# ---- validate_order 校验 ----
def test_validate_order_ok():
    o = validate_order(_order())
    assert o["side"] == "BUY" and o["qty"] == 100 and o["price"] == 10.0


@pytest.mark.parametrize("bad", [
    {"symbol": ""}, {"symbol": 123}, {"side": "HOLD"},
    {"qty": 0}, {"qty": -5}, {"qty": math.nan}, {"price": 0}, {"price": math.inf},
])
def test_validate_order_rejects(bad):
    with pytest.raises(ValueError):
        validate_order(_order(**bad))


# ---- pre_trade_check 隐性 KeyError 修复 ----
def test_pre_trade_no_keyerror_when_limit_keys_missing():
    # 不提供 max_position_pct / max_gross_pct 也不应抛 KeyError
    res = pre_trade_check(_order(), positions={}, limits={"gross": 1000})
    assert res["ok"] is True
    assert "notional" in res and "position_after" in res


def test_pre_trade_position_limits():
    limits = {"gross": 1000, "max_position_pct": 0.5, "max_gross_pct": 0.8}
    # 名义=100*10=1000 > 0.5*1000=500 -> 单标的超限
    res = pre_trade_check(_order(qty=100, price=10), positions={}, limits=limits)
    assert res["ok"] is False
    assert "单标的敞口超限" in res["reasons"]


def test_pre_trade_positions_aware():
    # 已有 50 股，再 BUY 100 -> position_after=150
    res = pre_trade_check(_order(side="BUY", qty=100), positions={"AAPL": 50}, limits={})
    assert res["position_after"] == 150
    # SELL 减仓：已有 200，SELL 100 -> position_after=100
    res = pre_trade_check(_order(side="SELL", qty=100), positions={"AAPL": 200}, limits={})
    assert res["position_after"] == 100


def test_pre_trade_order_notional_cap():
    limits = {"max_order_notional": 500}
    # 名义 100*10=1000 > 500
    res = pre_trade_check(_order(qty=100, price=10), positions={}, limits=limits)
    assert res["ok"] is False
    assert "单笔名义金额超限" in res["reasons"]


def test_pre_trade_invalid_order_propagates():
    with pytest.raises(ValueError):
        pre_trade_check({"symbol": "X", "side": "BUY", "qty": -1, "price": 1}, {}, {})


# ---- liquidity_check 新增能力 ----
def test_liquidity_check_pass():
    res = liquidity_check(_order(qty=100), adv=10000, max_participation=0.1)
    assert res["ok"] is True
    assert abs(res["participation"] - 0.01) < 1e-9


def test_liquidity_check_exceeds():
    res = liquidity_check(_order(qty=5000), adv=10000, max_participation=0.1)
    assert res["ok"] is False
    assert "流动性" in res["reasons"][0]


def test_liquidity_check_bad_adv():
    with pytest.raises(ValueError):
        liquidity_check(_order(), adv=0, max_participation=0.1)
