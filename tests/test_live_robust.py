"""LiveEngine 健壮性回归：外部依赖失败隔离 + None 现金回退。

实盘关键保障：connect / get_account 任一环节失败都不应让整个调仓循环崩溃，
而应降级返回带 errors 的状态快照；柜台未返回现金字段(None)时回退初始资金。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from lianghua.core.assets import AssetType
from lianghua.execution.brokers.base import BaseBroker, Order
from lianghua.execution.live import LiveEngine
from lianghua.execution.order_book import OrderBook


class StubGW:
    def fetch(self, *a, **k):
        return pd.DataFrame({"close": [10.0, 10.5, 11.0, 10.8, 11.2]})

    def fetch_minute(self, *a, **k):
        return self.fetch(*a, **k)

    def live_quote(self, *a, **k):
        return {"price": 11.0}


class LongStrat:
    def generate_signals(self, df):
        return pd.Series(1, index=df.index)


class BrokerRaisesConnect(BaseBroker):
    def connect(self):
        raise RuntimeError("柜台连接失败")

    def submit(self, o):
        return {"order_id": 1, "status": "filled", "fill_price": o.price}

    def get_account(self, mv=None):
        return {"cash": self.cash, "positions": {}}


class BrokerRaisesAccount(BaseBroker):
    def submit(self, o):
        return {"order_id": 1, "status": "filled", "fill_price": o.price}

    def get_account(self, mv=None):
        raise RuntimeError("账户查询超时")


class BrokerNoneCash(BaseBroker):
    def submit(self, o):
        return {"order_id": 1, "status": "filled", "fill_price": o.price}

    def get_account(self, mv=None):
        return {"cash": None, "positions": {}}  # 柜台未返回现金字段


def _mk(broker):
    db = tempfile.mktemp(suffix=".db")
    eng = LiveEngine(broker, StubGW(), LongStrat(), ["600519.SH"],
                     capital=1e6, order_db=db,
                     risk_limits={"max_order_value": 1e9})
    return eng, db


def test_connect_failure_degraded():
    eng, db = _mk(BrokerRaisesConnect())
    st = eng.step()
    assert st["errors"] and "connect" in st["errors"][0]
    os.remove(db)


def test_get_account_failure_degraded():
    eng, db = _mk(BrokerRaisesAccount())
    st = eng.step()
    assert st["errors"] and "get_account" in st["errors"][0]
    os.remove(db)


def test_none_cash_fallback():
    # 柜台未返回现金(None)时：绝不能崩溃；未知购买力应保守拒买，而非盲目下单。
    eng, db = _mk(BrokerNoneCash())
    st = eng.step()
    assert st["errors"] == [], "None 现金不应产生引擎错误: %s" % st["errors"]
    rej = [a for a in st["actions"] if a["action"] == "reject"]
    assert rej and "资金不足" in rej[0]["reason"], "未知现金应安全拒买: %s" % st["actions"]
    os.remove(db)
