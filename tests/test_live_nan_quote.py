# -*- coding: utf-8 -*-
"""坏报价（NaN / 0 / 异常）不得静默污染实盘引擎。

背景：``DataGateway.live_quote`` 在实时源与日线双双失败时返回 ``price=NaN``。
NaN 会安静地穿过所有朴素校验——``NaN or 0.0`` 仍是 NaN（NaN 为真值）、
``NaN <= 0`` 为 False——于是它一路流进两个要命的地方：

1. 市值 ``qty * NaN`` -> ``get_account`` 算出的整个账户权益变成 NaN，
   一个数字都不能看，还看不出是哪只标的坏的；
2. 止损/止盈的所有比较对 NaN 都是 False -> **持仓永远不会被止损**，
   而 ``_check_exits`` 只是 ``continue``，调用方以为止损仍在生效。

这两条都属于"不报错但有害"：用户看不到任何异常，却在拿错误的权益和失效的
风控做决策。本模块把它们钉死。
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.core.assets import AssetType
from lianghua.execution.brokers.base import BaseBroker, Position
from lianghua.execution.live import LiveEngine, _usable_price


class _Broker(BaseBroker):
    def __init__(self, positions=None, init_cash=1_000_000.0):
        super().__init__(init_cash=init_cash)
        self.positions = dict(positions or {})
        self.seen_mv = None

    def submit(self, order):
        self._oid += 1
        return {"order_id": self._oid, "status": "filled", "symbol": order.symbol,
                "side": order.side, "qty": order.qty, "price": order.price,
                "fill_price": order.price, "broker": "fake"}

    def get_account(self, market_values=None):
        self.seen_mv = dict(market_values or {})
        pos_value = sum(self.seen_mv.values())
        return {"cash": self.cash, "position_value": pos_value,
                "margin_frozen": 0.0, "equity": self.cash + pos_value,
                "positions": {s: getattr(p, "qty", 0)
                              for s, p in self.positions.items()}}


class _Gateway:
    """报价整体不可用的 gateway：live_quote 返回 NaN（与真实 source=none 一致）。"""

    def __init__(self, price=float("nan")):
        self.price = price

    def fetch(self, *a, **k):
        return pd.DataFrame({"close": np.linspace(10, 12, 30)})

    def fetch_minute(self, *a, **k):
        return pd.DataFrame({"close": np.linspace(10, 12, 30)})

    def live_quote(self, sym, asset=None):
        return {"symbol": sym, "price": self.price, "source": "none"}


class _Flat:
    def generate_signals(self, df):
        return pd.Series([0] * len(df))


def _engine(gateway=None, **kw):
    pos = Position("600519.SH", AssetType.STOCK, 100, 100.0)
    broker = _Broker(positions={"600519.SH": pos})
    eng = LiveEngine(broker, gateway or _Gateway(), _Flat(),
                     ["600000.SH"], **kw)  # 交易标的与持仓标的故意不同
    eng._last_acct = broker.get_account({})
    return eng, broker


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0, None, "abc"])
def test_usable_price_rejects_every_unusable_form(bad):
    assert _usable_price(bad) is None


def test_usable_price_accepts_finite_positive():
    assert _usable_price("12.5") == 12.5
    assert _usable_price(3) == 3.0


def test_nan_quote_never_reaches_market_value():
    """NaN 报价不得进入市值：整个账户权益不能被一条坏报价污染成 NaN。"""
    eng, broker = _engine()
    st = eng.step()
    assert broker.seen_mv, "应当对持仓做过市值标记"
    assert all(np.isfinite(v) for v in broker.seen_mv.values()), \
        "市值出现 NaN/inf：一条坏报价会让整个账户权益不可用"
    assert np.isfinite(st["account"].get("equity", float("nan")))
    # 回退口径必须是成本价 100 * 100 股
    assert broker.seen_mv["600519.SH"] == pytest.approx(10000.0)


def test_unavailable_quote_is_reported_not_swallowed():
    """取不到价必须写进 errors：按成本价标记是降级，不能装作一切正常。"""
    eng, _ = _engine()
    st = eng.step()
    assert any("600519.SH" in e and "成本价" in e for e in st["errors"]), \
        "持仓取价失败被静默吞掉，用户会把成本价当成市价"


def test_nan_quote_does_not_silently_disable_stop_loss():
    """报价为 NaN 时止损无法执行，必须显式上报，而不是 continue 了事。"""
    eng, _ = _engine(sl_pct=0.03, tp_pct=0.06)
    acts = eng._check_exits({"600519.SH": float("nan")})
    assert any(a.get("action") == "exit_skip" for a in acts), \
        "报价为 NaN 时止损静默失效且无任何提示"
    assert not any(a.get("action") == "exit" for a in acts)


def test_good_quote_still_triggers_stop_loss():
    """回归：正常报价下止损行为不变。"""
    eng, _ = _engine(gateway=_Gateway(price=96.0), sl_pct=0.03, tp_pct=0.06)
    acts = eng._check_exits({"600519.SH": 96.0})
    assert any(a.get("hit") == "STOP_LOSS" for a in acts)


def test_realtime_price_fallback_is_visible():
    """盘中模式实时价拿不到而回退到分钟线收盘，必须在 errors 里留痕。"""
    eng, _ = _engine(freq="1min")
    st = eng.step()
    assert any("决策价已回退" in e for e in st["errors"]), \
        "决策价从实时降级到收盘价是重要信息，不能无声发生"
