"""SimBroker 撮合正确性 + 额度守卫 + 可买数量测算测试。"""
import pytest

from lianghua.core.assets import AssetType
from lianghua.execution.brokers.base import Order
from lianghua.execution.brokers.sim import SimBroker


def test_stock_buy_then_sell_cash_flow():
    b = SimBroker(init_cash=100_000)
    r1 = b.submit(Order("600519.SH", "BUY", 100, 100.0))
    assert r1["status"] == "filled"
    assert r1["cash_after"] == 90_000.0
    r2 = b.submit(Order("600519.SH", "SELL", 100, 110.0))
    assert r2["status"] == "filled"
    assert r2["cash_after"] == 101_000.0
    assert b.get_account()["positions"]["600519.SH"] == 0


def test_validate_order_rejects_bad_input():
    b = SimBroker()
    assert b.submit(Order("600519.SH", "BUY", 0, 100.0))["status"] == "rejected"
    assert b.submit(Order("600519.SH", "BUY", 100, -1.0))["status"] == "rejected"
    assert b.submit(Order("600519.SH", "HOLD", 100, 100.0))["status"] == "rejected"


def test_insufficient_cash_rejected():
    b = SimBroker(init_cash=1_000)
    r = b.submit(Order("600519.SH", "BUY", 100, 100.0))  # 需 10000
    assert r["status"] == "rejected"
    assert r["reason"] == "现金/保证金不足"
    assert b.cash == 1_000  # 未改动


def test_future_short_frees_no_cash_margin_bugfix():
    # 期货开多/开空均冻结保证金，空头不得把保证金存入现金
    b = SimBroker(init_cash=1_000_000)
    b.submit(Order("RB2410.SHF", "BUY", 1, 3000.0, AssetType.FUTURE))
    assert b.cash == 997_000.0  # 1*3000*10*0.1 = 3000 冻结
    b.submit(Order("RB2410.SHF", "SELL", 1, 3000.0, AssetType.FUTURE))
    # 关闭仓位，两次均冻结 3000 -> 1,000,000 - 6000
    assert b.cash == 994_000.0
    assert b.get_account()["positions"]["RB2410.SHF"] == 0


def test_future_margin_insufficient_rejected():
    b = SimBroker(init_cash=1_000)
    r = b.submit(Order("RB2410.SHF", "BUY", 10, 3000.0, AssetType.FUTURE))  # 需 30000
    assert r["status"] == "rejected"
    assert b.cash == 1_000


def test_max_affordable():
    b = SimBroker(init_cash=10_000)
    assert b.max_affordable("600519.SH", 100.0, "stock") == 100
    assert b.max_affordable("RB2410.SHF", 3000.0, "future") == 3  # 每手 3000
    assert b.max_affordable("510050C3000.SH", 0.05, "option") == 20  # 每手 500
    assert b.max_affordable("600519.SH", 0.0, "stock") == 0  # 非法价格


def test_return_rate_observable():
    b = SimBroker(init_cash=100_000)
    b.submit(Order("600519.SH", "BUY", 100, 100.0))
    b.submit(Order("600519.SH", "SELL", 100, 110.0))
    acct = b.get_account()
    assert "return_rate" in acct
    # 10万本金，买100@100 卖100@110 -> 盈利 1000 -> 收益率 1%
    assert acct["return_rate"] == pytest.approx(0.01)


def test_option_buy_pays_premium_sell_receives():
    b = SimBroker(init_cash=1_000_000)
    r = b.submit(Order("510050C3000.SH", "BUY", 10, 0.05, AssetType.OPTION))
    # 10 * 0.05 * 10000 = 5000 付权利金
    assert r["status"] == "filled"
    assert b.cash == 995_000.0
    r2 = b.submit(Order("510050C3000.SH", "SELL", 10, 0.06, AssetType.OPTION))
    # 10 * 0.06 * 10000 = 6000 收权利金
    assert r2["status"] == "filled"
    assert b.cash == 1_001_000.0
