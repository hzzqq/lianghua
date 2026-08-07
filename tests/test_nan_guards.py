"""非有限值（NaN/inf）穿透审计：把上一轮在 live.py 挖出的那**一类**缺陷查干净。

背景：`execution/live.py` 曾因 `price or 0.0`（NaN 为真值，结果仍是 NaN）与
`price <= 0`（对 NaN 恒为 False）两个朴素守卫被 NaN 报价穿透，导致账户权益变成
NaN、且所有止损比较恒为 False —— **持仓永远不会被止损，还一声不吭**。

这是一整类缺陷而非孤例：Python 中 `x or y`、`x <= 0`、`x > y` 在 NaN 面前全部
静默失效。本文件对全仓机械扫描出的候选点逐一构造证据，覆盖三种后果：

1. **fail-open（闸门失守）**：风控/盘前检查对 NaN 输入返回"通过"，等于没有风控。
2. **静默失效**：止损/止盈判定对 NaN 恒为 False，调用方以为保护还在。
3. **状态腐化**：inf 行情把追踪止损的 stop_price 永久污染，之后每 tick 误触发。

统一策略：
- 有安全出口的**闸门类**接口（返回 ok / size）→ fail-closed，返回拒绝/0。
- 无法表达"未知"的**判定类**接口（返回 bool / list）→ 显式抛 ValueError，
  绝不用一个看起来正常的返回值掩盖"我算不出来"。
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest

from lianghua.core.assets import AssetType
from lianghua.core.numeric import is_finite_num, positive_finite
from lianghua.execution.brokers.base import Order
from lianghua.execution.order_book import OrderBook
from lianghua.execution.orders import (bracket_order, evaluate_bracket,
                                       evaluate_oco, evaluate_trailing_stop,
                                       oco_order, trailing_stop_order)
from lianghua.execution.pre_trade import pre_trade_check
from lianghua.risk.manager import RiskManager

NAN = float("nan")
INF = float("inf")
BAD = [NAN, INF, -INF]


# ---------------------------------------------------------------- 共享工具
def test_numeric_helpers_reject_non_finite():
    """共享谓词本身必须先站得住，否则下游守卫全是空中楼阁。"""
    for v in BAD:
        assert is_finite_num(v) is False
        assert positive_finite(v) is None
    assert is_finite_num(0.0) and is_finite_num(-3) and is_finite_num("1.5")
    assert is_finite_num(None) is False
    assert is_finite_num("abc") is False
    assert positive_finite(10.0) == 10.0
    assert positive_finite(0.0) is None      # 0 不是可交易价格
    assert positive_finite(-1.0) is None


# ------------------------------------------------ 1. RiskManager 闸门失守
def test_risk_position_size_rejects_non_finite_price():
    """NaN 价格穿过 `price <= 0`，风控返回满仓比例 —— 拿 NaN 去算下单量。"""
    rm = RiskManager()
    for bad in BAD:
        assert rm.position_size(1_000_000.0, bad) == 0.0, (
            f"price={bad} 仍返回可部署比例，等于用坏价格决定仓位")


def test_risk_position_size_rejects_non_finite_cash():
    rm = RiskManager()
    for bad in BAD:
        assert rm.position_size(bad, 10.0) == 0.0, f"cash={bad} 仍允许建仓"


def test_risk_check_entry_rejects_non_finite_cash():
    """`cash <= 0` 对 NaN 恒为 False，于是 NaN/inf 现金一路放行开新仓。"""
    rm = RiskManager()
    for bad in BAD:
        assert rm.check_entry("600519.SH", bad) is False, f"cash={bad} 被放行"


def test_risk_check_exit_raises_on_non_finite_price():
    """止损判定拿不到可信价格时必须响亮报错，而不是返回 False 假装"无需止损"。"""
    rm = RiskManager(stop_loss=0.10)
    assert rm.check_exit(100.0, 85.0) is True      # 正常路径不受影响
    assert rm.check_exit(100.0, 95.0) is False
    for bad in BAD:
        with pytest.raises(ValueError):
            rm.check_exit(100.0, bad)
        with pytest.raises(ValueError):
            rm.check_exit(bad, 95.0)


def test_risk_check_margin_and_premium_reject_non_finite():
    """保证金/权利金闸门同样不能被 NaN 蒙混（fail-closed）。"""
    rm = RiskManager()
    for bad in BAD:
        assert rm.check_margin(bad, 1e6, AssetType.FUTURE) is False
        assert rm.check_margin(1e5, bad, AssetType.FUTURE) is False
        assert rm.premium_ok(bad, 1e6, AssetType.OPTION) is False
        assert rm.premium_ok(1e4, bad, AssetType.OPTION) is False
    # 非目标资产类别仍旧直接放行，行为不变
    assert rm.check_margin(NAN, NAN, AssetType.STOCK) is True
    assert rm.premium_ok(NAN, NAN, AssetType.STOCK) is True


# --------------------------------------------- 2. pre_trade_check 闸门失守
def _acct(cash=1e9, positions=None):
    return {"cash": cash, "positions": positions or {}}


def test_pre_trade_rejects_non_finite_account_cash():
    """`notional > cash` 对 NaN 现金恒为 False —— 资金充足检查形同虚设。

    这是 LiveEngine 落单前的最后一道闸门（live.py 两处调用），放行即真下单。
    """
    order = Order(symbol="600000", side="BUY", qty=100, price=10.0)
    for bad in BAD:
        ok, reason = pre_trade_check(order, _acct(cash=bad))
        assert ok is False, f"cash={bad} 通过了资金检查"
        assert "现金" in reason or "资金" in reason


def test_pre_trade_rejects_non_finite_qty():
    """qty=NaN 会让 min_qty / 名义金额 / 资金三道检查全部恒 False 通过。"""
    for bad in BAD:
        ok, _ = pre_trade_check(Order("600000", "BUY", bad, 10.0), _acct())
        assert ok is False, f"qty={bad} 全程通过盘前风控"


def test_pre_trade_rejects_non_finite_position():
    """持仓数量为 NaN 时，持仓上限检查静默通过。"""
    order = Order(symbol="600000", side="BUY", qty=100, price=10.0)
    ok, _ = pre_trade_check(order, _acct(positions={"600000": NAN}),
                            {"max_position_value": 500.0})
    assert ok is False


def test_pre_trade_normal_path_unchanged():
    """守卫不得误伤正常单：足额现金 + 合法数量必须照常通过。"""
    ok, reason = pre_trade_check(Order("600000", "BUY", 100, 10.0), _acct())
    assert ok is True, reason
    ok2, _ = pre_trade_check(Order("600000", "SELL", 100, 10.0),
                             _acct(cash=0.0, positions={"600000": 100}))
    assert ok2 is True          # 卖出不查现金
    ok3, _ = pre_trade_check(Order("600000", "BUY", 100, 200.0),
                             _acct(cash=1000.0))
    assert ok3 is False         # 真·资金不足仍要拦


# ------------------------------------------ 3. 高级订单：静默失效 + 状态腐化
def test_evaluate_bracket_raises_on_non_finite_price():
    """行情为 NaN 时括号单三条腿一条都不触发，止损静默失效。"""
    b = bracket_order("600519.SH", "BUY", 100, 100.0, 95.0, 110.0)
    assert evaluate_bracket(b, 94.0) == ["stop"]     # 正常路径不变
    for bad in BAD:
        with pytest.raises(ValueError):
            evaluate_bracket(b, bad)


def test_evaluate_oco_raises_on_non_finite_price():
    o = oco_order("600519.SH", "BUY", 100, 90.0, 110.0)
    assert evaluate_oco(o, 89.0) == ["a"]
    for bad in BAD:
        with pytest.raises(ValueError):
            evaluate_oco(o, bad)


def test_trailing_stop_inf_price_corrupts_state():
    """inf 行情让 `new_stop > stop_price` 成立，stop_price 被永久写成 inf。

    之后每一次评估都返回一张 price=inf 的平仓单 —— 状态腐化且不可自愈。
    """
    st = trailing_stop_order("600519.SH", "BUY", 100, 100.0, 0.05)
    assert evaluate_trailing_stop(st, 100.0) is None
    assert st["stop_price"] == pytest.approx(95.0)
    with pytest.raises(ValueError):
        evaluate_trailing_stop(st, INF)
    assert math.isfinite(st["stop_price"]), "非法行情把 stop_price 污染成非有限值"
    assert st["stop_price"] == pytest.approx(95.0), "非法行情不得改写止损价"
    with pytest.raises(ValueError):
        evaluate_trailing_stop(st, NAN)
    assert st["stop_price"] == pytest.approx(95.0)
    # 正常路径仍旧工作：涨到 110 -> 止损上移到 104.5，跌破即触发
    assert evaluate_trailing_stop(st, 110.0) is None
    assert st["stop_price"] == pytest.approx(104.5)
    order = evaluate_trailing_stop(st, 100.0)
    assert order is not None and order.side == "BUY"


def test_trailing_stop_order_rejects_non_finite_activation():
    for bad in BAD:
        with pytest.raises(ValueError):
            trailing_stop_order("X", "BUY", 100, bad, 0.05)
        with pytest.raises(ValueError):
            trailing_stop_order("X", "BUY", 100, 100.0, bad)


def test_bracket_order_rejects_non_finite_levels():
    """`stop < entry < target` 对 NaN 恒 False，靠 `not(...)` 侥幸拦住；显式化。"""
    for bad in BAD:
        with pytest.raises(ValueError):
            bracket_order("X", "BUY", 100, bad, 95.0, 110.0)
        with pytest.raises(ValueError):
            bracket_order("X", "BUY", 100, 100.0, bad, 110.0)


# ------------------------------------------------- 4. 订单账本落库脏成交价
def test_order_book_rejects_non_finite_fill_price():
    """record() 校验了 price 却漏了 fill_price，NaN 成交价可直接落库。

    账本是对账与绩效统计的唯一事实来源，一条 NaN 会污染整段统计。
    """
    path = os.path.join(tempfile.mkdtemp(), "ob_nan.db")
    try:
        book = OrderBook(path)
        for bad in BAD:
            with pytest.raises(ValueError):
                book.record("600000", "BUY", 100, 10.0, fill_price=bad)
        oid = book.record("600000", "BUY", 100, 10.0)     # 默认 0.0 仍合法
        assert oid > 0
        with pytest.raises(ValueError):
            book.fill(oid, NAN)
        book.fill(oid, 10.5)
        assert book.all()[0]["fill_price"] == 10.5
    finally:
        if os.path.exists(path):
            os.remove(path)


# -------------------------------- 5. LiveEngine 残留的 `x or y` 兜底穿透
class _Pos:
    def __init__(self, qty, avg_price, asset_type=AssetType.STOCK):
        self.qty = qty
        self.avg_price = avg_price
        self.asset_type = asset_type


class _Broker:
    def __init__(self, positions):
        self.positions = positions

    def connect(self):
        return True

    def submit(self, order):
        return {"status": "filled", "fill_price": NAN}   # 柜台回报脏成交价

    def get_account(self, mv=None):
        return {"cash": 100000.0, "positions": {}}


class _GW:
    def live_quote(self, *a, **k):
        return {"price": NAN}


def _engine(positions, tmpdir, **kw):
    from lianghua.execution.live import LiveEngine
    return LiveEngine(_Broker(positions), _GW(), lambda df: None,
                      ["600519.SH"], order_db=os.path.join(tmpdir, "lo.db"),
                      **kw)


def test_check_exits_nan_cost_basis_no_longer_silent(tmp_path):
    """`avg = float(pos.avg_price or px)`：NaN 成本价穿过 `or`，止损再次静默失效。

    上一轮修的是"报价"侧，同一函数里"成本价"侧的同类兜底被漏掉了。
    """
    eng = _engine({"600519.SH": _Pos(100, NAN)}, str(tmp_path), sl_pct=0.03)
    acts = eng._check_exits({"600519.SH": 50.0})     # 报价正常，成本价是 NaN
    assert acts, "成本价不可用时必须上报，绝不能静默跳过止损"
    a = acts[0]
    assert a["action"] in ("exit", "exit_skip"), a
    if a["action"] == "exit_skip":
        assert "成本" in a["reason"] or "均价" in a["reason"], a["reason"]


def test_check_exits_zero_cost_basis_still_falls_back(tmp_path):
    """avg_price=0（未初始化）仍按原语义回退到当前价，不误报。"""
    eng = _engine({"600519.SH": _Pos(100, 0.0)}, str(tmp_path), sl_pct=0.03)
    acts = eng._check_exits({"600519.SH": 50.0})
    assert acts == [] or acts[0]["action"] == "hold", acts


def test_step_never_records_non_finite_fill_price(tmp_path):
    """柜台回报 fill_price=NaN 时，`res.get(...) or px` 仍取到 NaN 并写进账本。"""
    from lianghua.execution.live import LiveEngine
    eng = _engine({"600519.SH": _Pos(100, 40.0)}, str(tmp_path), sl_pct=0.03)
    assert isinstance(eng, LiveEngine)
    acts = eng._check_exits({"600519.SH": 30.0})     # -25%，必定触发止损
    exits = [a for a in acts if a["action"] == "exit"]
    assert exits, acts
    fp = exits[0]["fill_price"]
    assert math.isfinite(fp), f"NaN 成交价被当成真实成交价记账: {fp}"
    assert fp == pytest.approx(30.0), "取不到有效回报价时应退回决策价"
