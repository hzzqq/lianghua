# -*- coding: utf-8 -*-
"""第十二轮「连接实盘」测试：LiveEngine / OrderBook / QMT / PTrade / live_quote。

覆盖：
1. OrderBook SQLite 持久化（record/update/pending/recent/clear）
2. LiveEngine + FakeBroker 单步调仓（信号→风控→下单→账本）
3. make_live_engine paper 模式端到端（演示数据回退，零网络依赖）
4. 真实柜台双保险：qmt/pt 不带 live=True 直接拒绝
5. QmtBroker / PtBroker 无 SDK 时 connect() 优雅报错
6. DataGateway.live_quote 断网降级（回退最近收盘价）
7. 紧急停止 kill()：后续 step 直接跳过
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from lianghua.core.assets import AssetType
from lianghua.execution.brokers import make_broker
from lianghua.execution.brokers.base import BaseBroker, Order
from lianghua.execution.brokers.pt import PtBroker
from lianghua.execution.brokers.qmt import QmtBroker
from lianghua.execution.live import LiveEngine, make_live_engine
from lianghua.execution.order_book import OrderBook

PASS = []


def ok(name):
    PASS.append(name)
    print("  ✓", name)


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


# ---------- 1. OrderBook ----------
def test_order_book():
    db = _tmp_db()
    book = OrderBook(db)
    oid = book.record("600519.SH", "BUY", 100, 1700.0, asset="stock",
                      broker="PaperBroker")
    assert oid >= 1
    assert len(book.pending()) == 1
    book.update(oid, status="filled", fill_price=1700.5)
    rows = book.recent(5)
    assert rows[0]["status"] == "filled" and abs(rows[0]["fill_price"] - 1700.5) < 1e-9
    assert len(book.all()) == 1
    # 跨实例持久化
    book2 = OrderBook(db)
    assert len(book2.all()) == 1
    book2.clear()
    assert len(book2.all()) == 0
    os.remove(db)
    ok("OrderBook 持久化 record/update/pending/recent/clear")


# ---------- 2. LiveEngine + FakeBroker ----------
class FakeBroker(BaseBroker):
    """始终成交的桩 broker，用于验证引擎胶水。"""

    def __init__(self, init_cash=1_000_000.0):
        super().__init__(init_cash=init_cash)
        self.submitted = []

    def submit(self, order: Order) -> dict:
        self.submitted.append(order)
        cost = order.qty * order.price
        self.cash += -cost if order.side == "BUY" else cost
        return {"order_id": self._next_id(), "status": "filled",
                "fill_price": order.price}

    def get_account(self, market_values=None) -> dict:
        return {"cash": self.cash, "position_value": 0.0, "margin_frozen": 0.0,
                "equity": self.cash, "positions": {}}


class AlwaysLong:
    """恒为 1 的策略桩。"""

    def generate_signals(self, df):
        return pd.Series(1, index=df.index)


def test_live_engine_fake():
    from lianghua.data.gateway import DataGateway
    db = _tmp_db()
    broker = FakeBroker()
    eng = LiveEngine(broker, DataGateway(), AlwaysLong(), ["600519.SH"],
                     capital=1e6, order_db=db,
                     risk_limits={"max_order_value": 1e9})
    st = eng.step()
    assert st["mode"] == "paper" and not st["killed"]
    trades = [a for a in st["actions"] if a["action"] == "trade"]
    assert trades, "AlwaysLong 应产生买入动作: %s" % st["actions"]
    assert broker.submitted and broker.submitted[0].side == "BUY"
    assert broker.submitted[0].qty % 100 == 0  # 股票按手
    assert len(eng.book.all()) == len(trades)
    os.remove(db)
    ok("LiveEngine+FakeBroker 单步：信号→风控→下单→账本")


def test_live_engine_risk_reject():
    from lianghua.data.gateway import DataGateway
    db = _tmp_db()
    eng = LiveEngine(FakeBroker(), DataGateway(), AlwaysLong(), ["600519.SH"],
                     capital=1e6, order_db=db,
                     risk_limits={"max_order_value": 1.0})  # 单笔上限 1 元
    st = eng.step()
    rejects = [a for a in st["actions"] if a["action"] == "reject"]
    assert rejects and "上限" in rejects[0]["reason"]
    assert len(eng.book.all()) == 0  # 拒单不进账本
    os.remove(db)
    ok("盘前风控拒单（单笔上限）且不入账本")


def test_kill_switch():
    from lianghua.data.gateway import DataGateway
    db = _tmp_db()
    eng = LiveEngine(FakeBroker(), DataGateway(), AlwaysLong(), ["600519.SH"],
                     order_db=db)
    eng.kill()
    st = eng.step()
    assert st["killed"] and st["actions"] == []
    assert eng.run(max_steps=3) == []
    os.remove(db)
    ok("紧急停止 kill()：step/run 直接跳过")


# ---------- 3. paper 端到端 ----------
def test_paper_end_to_end():
    db = _tmp_db()
    eng = make_live_engine("paper", "sma_cross",
                           ["600519.SH", "000300.SH"], capital=1e6)
    eng.book = OrderBook(db)  # 隔离测试账本
    out = eng.run(max_steps=2)
    assert len(out) == 2
    for st in out:
        assert st["mode"] == "paper"
        assert "cash" in st["account"] and "equity" in st["account"]
        assert set(st["quotes"]) <= {"600519.SH", "000300.SH"}
    os.remove(db)
    ok("make_live_engine paper 端到端 2 步（演示数据回退）")


# ---------- 4. 真实柜台双保险 ----------
def test_live_guard():
    for kind in ("qmt", "pt"):
        try:
            make_live_engine(kind, "sma_cross", ["600519.SH"])
            raise AssertionError("%s 不带 live=True 应被拒绝" % kind)
        except RuntimeError as e:
            assert "live=True" in str(e)
    ok("真实柜台双保险：qmt/pt 缺 live=True 拒绝创建")


# ---------- 5. SDK 缺失优雅报错 ----------
def test_qmt_pt_graceful():
    b = QmtBroker(account_id="test")
    try:
        b.connect()
        raise AssertionError("无 xtquant 应报错")
    except RuntimeError as e:
        assert "xtquant" in str(e)
    try:
        QmtBroker().connect()  # 无 account_id
        raise AssertionError("无 account_id 应报错")
    except RuntimeError as e:
        assert "account_id" in str(e)
    try:
        PtBroker().connect()
        raise AssertionError("无 ptrade 应报错")
    except RuntimeError as e:
        assert "ptrade" in str(e)
    assert isinstance(make_broker("qmt"), QmtBroker)
    assert isinstance(make_broker("pt"), PtBroker)
    ok("QMT/PTrade 无 SDK 时 connect() 优雅 RuntimeError")


# ---------- 6. PtBroker 注入桩 ----------
class _PtStub:
    class _Style:
        def __init__(self, p):
            self.price = p

    def LimitOrder(self, p):
        return self._Style(p)

    def order(self, code, amount, style=None):
        self.last = (code, amount, getattr(style, "price", None))
        return "pt-001"

    def get_asset(self):
        return {"cash": 88888.0}

    def get_positions(self):
        return []


def test_pt_with_stub():
    api = _PtStub()
    b = PtBroker(account_id="acc", api=api)
    assert b.connect()
    res = b.submit(Order("600519.SH", "SELL", 200, 1690.0,
                         asset_type=AssetType.STOCK))
    assert res["status"] == "submitted" and res["order_id"] == "pt-001"
    assert api.last == ("600519.SH", -200, 1690.0)  # 卖出=负 amount
    acct = b.get_account()
    assert acct["cash"] == 88888.0
    ok("PtBroker 注入桩：order 映射/负 amount 卖出/get_account")


# ---------- 7. live_quote 降级 ----------
def test_live_quote():
    from lianghua.data.gateway import DataGateway
    q = DataGateway().live_quote("600519.SH")
    assert q["symbol"] == "600519.SH" and q["price"] > 0
    assert "source" in q
    ok("live_quote 实时/降级快照 price>0（source=%s）" % q["source"])


if __name__ == "__main__":
    print("=== 第十二轮 连接实盘 测试 ===")
    test_order_book()
    test_live_engine_fake()
    test_live_engine_risk_reject()
    test_kill_switch()
    test_paper_end_to_end()
    test_live_guard()
    test_qmt_pt_graceful()
    test_pt_with_stub()
    test_live_quote()
    print("\n全部 %d 项实盘测试通过 ✅" % len(PASS))
