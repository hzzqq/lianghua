"""execution/ 打磨：订单账本守卫与便捷方法、盘前风控价格/方向守卫、账户路由。"""
import os
import tempfile

import pytest

from lianghua.execution.order_book import OrderBook
from lianghua.execution.pre_trade import pre_trade_check
from lianghua.execution.router import AccountRouter
from lianghua.execution.brokers.base import Order


def _book():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return OrderBook(path), path


def test_record_validates_side_qty_price():
    book, path = _book()
    try:
        with pytest.raises(ValueError):
            book.record("600000", "HOLD", 100, 10.0)   # 非法 side
        with pytest.raises(ValueError):
            book.record("600000", "BUY", 0, 10.0)      # 非正 qty
        with pytest.raises(ValueError):
            book.record("600000", "BUY", 100, 0.0)     # 非正 price
        oid = book.record("600000", "buy", 100, 10.0)  # 小写归一
        assert oid >= 1
        pending = book.pending()
        assert pending[0]["side"] == "BUY"
    finally:
        os.remove(path)


def test_fill_and_summary():
    book, path = _book()
    try:
        oid = book.record("600000", "BUY", 100, 10.0)
        assert book.summary() == {"submitted": 1}
        book.fill(oid, 10.5)
        row = book.all()[0]
        assert row["status"] == "filled"
        assert row["fill_price"] == 10.5
        assert book.summary() == {"filled": 1}
    finally:
        os.remove(path)


def test_pre_trade_basic():
    order = Order(symbol="600000", side="BUY", qty=100, price=10.0)
    acct = {"cash": 10000.0, "positions": {}}
    ok, reason = pre_trade_check(order, acct)
    assert ok, reason


def test_pre_trade_insufficient_cash():
    order = Order(symbol="600000", side="BUY", qty=100, price=200.0)
    acct = {"cash": 1000.0, "positions": {}}
    ok, _ = pre_trade_check(order, acct)
    assert ok is False


def test_pre_trade_blocked_and_limits():
    order = Order(symbol="600000", side="BUY", qty=100, price=10.0)
    acct = {"cash": 1e9, "positions": {}}
    ok, _ = pre_trade_check(order, acct, {"blocked": ["600000"]})
    assert ok is False
    ok2, _ = pre_trade_check(order, acct, {"max_order_value": 500.0})
    assert ok2 is False


def test_pre_trade_rejects_bad_price_and_side():
    acct = {"cash": 1e9, "positions": {}}
    zp = Order(symbol="600000", side="BUY", qty=100, price=0.0)
    ok, _ = pre_trade_check(zp, acct)
    assert ok is False                       # 0 价绕过资金检查被拦下
    neg = Order(symbol="600000", side="BUY", qty=100, price=-5.0)
    ok2, _ = pre_trade_check(neg, acct)
    assert ok2 is False
    bad = Order(symbol="600000", side="XYZ", qty=100, price=10.0)
    ok3, _ = pre_trade_check(bad, acct)
    assert ok3 is False


def test_account_router_precedence():
    router = AccountRouter({
        "by_asset": {"stock": "A", "future": "B"},
        "by_prefix": {"600": "C"},
        "default": "D",
    }, default="D")
    assert router.route("600519", "stock") == "A"        # by_asset 命中
    assert router.route("IF2309", "future") == "B"       # by_asset 命中
    # by_asset 优先于 by_prefix：600000 属 stock -> A 而非前缀 C
    assert router.route("600000", "stock") == "A"
    # 资产未命中时回退到前缀匹配
    assert router.route("600000", "crypto") == "C"
    # 都未命中 -> 默认
    assert router.route("000001", "etf") == "D"
