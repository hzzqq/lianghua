# -*- coding: utf-8 -*-
"""第十四轮深化测试：盘中止损/止盈 + 多账户再平衡 + 微信推送。

覆盖：SL/TP 触发平仓并入账、正常价不触发、多账户再平衡计划、
WeChatNotifier 空 key 返回 False、build_notifier(None)、WeComAppNotifier 离线优雅返回。
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from lianghua.execution.brokers.base import BaseBroker, Position
from lianghua.execution.live import LiveEngine
from lianghua.execution.router import make_multi_engine
from lianghua.execution.notify import (WeChatNotifier, WeComAppNotifier,
                                        NotifyHub, build_notifier)
from lianghua.core.assets import AssetType


class FakeBroker(BaseBroker):
    """测试用 broker：持仓固定，submit 直接成交。"""

    def __init__(self, positions=None, init_cash=1_000_000.0):
        super().__init__(init_cash=init_cash)
        self.positions = dict(positions or {})

    def submit(self, order):
        self._oid += 1
        return {"order_id": self._oid, "status": "filled", "symbol": order.symbol,
                "side": order.side, "qty": order.qty, "price": order.price,
                "fill_price": order.price, "broker": "fake"}

    def get_account(self, market_values=None):
        pos_value = sum(abs(getattr(p, "qty", 0)) * getattr(p, "avg_price", 0)
                        for p in self.positions.values())
        return {"cash": self.cash, "position_value": pos_value,
                "margin_frozen": 0.0, "equity": self.cash + pos_value,
                "positions": {s: getattr(p, "qty", 0)
                              for s, p in self.positions.items()}}


class FakeGateway:
    def fetch(self, *a, **k):
        return pd.DataFrame({"close": np.linspace(10, 12, 30)})

    def fetch_minute(self, *a, **k):
        return pd.DataFrame({"close": np.linspace(10, 12, 30)})

    def live_quote(self, sym, asset=None):
        return {"symbol": sym, "price": 0.0, "source": "fake"}


class FakeStrategy:
    def generate_signals(self, df):
        return pd.Series([0] * len(df))


def _engine_with_pos(sl=None, tp=None, trail=None, price=100.0):
    pos = Position("600519.SH", AssetType.STOCK, 100, price)
    broker = FakeBroker(positions={"600519.SH": pos})
    eng = LiveEngine(broker, FakeGateway(), FakeStrategy(), ["600519.SH"],
                     sl_pct=sl, tp_pct=tp, trailing_pct=trail)
    eng._last_acct = broker.get_account({})
    return eng


def test_stop_loss_triggers_exit():
    eng = _engine_with_pos(sl=0.03, tp=0.06, trail=0.04, price=100.0)
    acts = eng._check_exits({"600519.SH": 96.0})  # -4% 跌破 -3%
    exits = [a for a in acts if a.get("action") == "exit"]
    assert exits, "应触发止损平仓"
    assert exits[0]["side"] == "SELL"
    assert exits[0]["hit"] == "STOP_LOSS"
    recs = eng.book.recent(5)
    assert any(r["side"] == "SELL" for r in recs), "账本应记录平仓"


def test_take_profit_triggers_exit():
    eng = _engine_with_pos(sl=0.03, tp=0.06, price=100.0)
    acts = eng._check_exits({"600519.SH": 107.0})  # +7% 超过 +6%
    assert any(a.get("hit") == "TAKE_PROFIT" for a in acts)


def test_trailing_stop_triggers_exit():
    eng = _engine_with_pos(sl=0.03, tp=0.06, trail=0.04, price=100.0)
    # 先抬升峰值到 110，再回落到 105（从峰值回撤 4.5% > 4%）
    eng._check_exits({"600519.SH": 110.0})
    acts = eng._check_exits({"600519.SH": 105.0})
    assert any(a.get("hit") == "TRAILING_STOP" for a in acts)


def test_no_exit_when_price_normal():
    eng = _engine_with_pos(sl=0.03, tp=0.06, price=100.0)
    acts = eng._check_exits({"600519.SH": 102.0})
    assert not any(a.get("action") == "exit" for a in acts)


def test_short_cover_on_stop():
    # 空头持仓，涨破 +3% 触发止损回补
    pos = Position("IF.CFE", AssetType.FUTURE, -1, 3000.0)
    broker = FakeBroker(positions={"IF.CFE": pos})
    eng = LiveEngine(broker, FakeGateway(), FakeStrategy(), ["IF.CFE"], sl_pct=0.03)
    eng._last_acct = broker.get_account({})
    acts = eng._check_exits({"IF.CFE": 3100.0})  # +3.3%
    exits = [a for a in acts if a.get("action") == "exit"]
    assert exits and exits[0]["side"] == "BUY", "空头止损应 BUY 回补"


def test_rebalance_plan():
    cfg = {
        "symbols": ["600519.SH", "IF.CFE"],
        "accounts": [
            {"name": "A", "broker": "paper", "strategy": "sma_cross",
             "capital": 1_000_000.0, "risk": {}},
            {"name": "B", "broker": "paper", "strategy": "sma_cross",
             "capital": 500_000.0, "risk": {}},
        ],
    }
    eng = make_multi_engine(cfg)
    plan = eng.rebalance({"A": 0.5, "B": 0.5})
    assert plan["total"] == 1_500_000.0
    deltas = sum(p["delta"] for p in plan["plan"])
    assert abs(deltas) < 1e-6, "transfer 计划合计应为 0"
    assert plan["plan"][0]["target_equity"] == 750_000.0


def test_wechat_empty_key_returns_false():
    assert WeChatNotifier("").push("x") is False
    assert WeChatNotifier(None).push("x") is False


def test_build_notifier_none():
    assert build_notifier(None) is None
    hub = build_notifier({"wechat_webhook": "693a91f6-demo"})
    assert isinstance(hub, NotifyHub)


def test_wecom_load_and_push_offline():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "wecom_config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"corpid": "wwx", "corpsecret": "s",
                   "agentid": 1, "touser": "黄子州"}, f)
    n = WeComAppNotifier(p)
    assert n.corpid == "wwx"
    assert n.push("hi") is False  # 沙箱无网，优雅返回 False 不抛


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
