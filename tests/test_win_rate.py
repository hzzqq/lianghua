"""round_trip_win_rate / round_trip_stats 的真实性测试。

回归守卫：旧 perf.metrics.win_rate 用相邻 cash_after 单调变化近似胜率，
在含止损平仓、资金约束导致现金曲线非单调时会失真。本测试确保
逐笔 BUY→SELL 配对口径正确，且止损平仓(SELL reason=stop_loss)被正确计入回合。
"""
from __future__ import annotations

from lianghua.perf.metrics import round_trip_win_rate, round_trip_stats, win_rate


def _trades():
    # 两笔完整回合：第 1 笔盈利，第 2 笔亏损（含止损平仓）
    return [
        {"side": "BUY",  "price": 100.0, "qty": 100, "fee": 30.0, "cash_after": 990000.0},
        {"side": "SELL", "price": 110.0, "qty": 100, "fee": 63.0, "reason": "signal",   "cash_after": 991007.0},
        {"side": "BUY",  "price": 200.0, "qty": 100, "fee": 60.0, "cash_after": 991000.0},
        {"side": "SELL", "price": 180.0, "qty": 100, "fee": 54.0, "reason": "stop_loss", "cash_after": 988886.0},
    ]


def test_round_trip_win_rate_basic():
    tr = _trades()
    assert round_trip_win_rate(tr) == 0.5           # 1 胜 1 负
    stats = round_trip_stats(tr)
    assert stats["n"] == 2
    # 盈利回合净收益 = 110*100 - 63 - (100*100 + 30) = 10937 - 10030 = 907
    # 亏损回合净收益 = 180*100 - 54 - (200*100 + 60) = 17946 - 20060 = -2114
    assert abs(stats["avg_win"] - 907.0) < 1e-6
    assert abs(stats["avg_loss"] - 2114.0) < 1e-6
    assert stats["profit_factor"] == 907.0 / 2114.0
    assert stats["total_pnl"] == 907.0 - 2114.0


def test_stop_loss_sell_is_counted():
    tr = _trades()
    # 两笔都是完整 BUY→SELL，止损平仓(SIDE=SELL reason=stop_loss)也被配对
    assert round_trip_stats(tr)["n"] == 2


def test_no_trades_returns_zero():
    assert round_trip_win_rate([]) == 0.0
    assert round_trip_stats([])["n"] == 0


def test_orphan_sell_ignored():
    # 无前置 BUY 的 SELL 不应产生孤儿回合（配对要求严格 BUY→SELL）
    tr = [{"side": "SELL", "price": 50.0, "qty": 100, "fee": 10.0}]
    assert round_trip_stats(tr)["n"] == 0


def test_deprecated_win_rate_still_runs():
    # 旧接口保留向后兼容：至少能跑且返回 [0,1] 区间（不要求数值正确）
    tr = _trades()
    w = win_rate(tr)
    assert 0.0 <= w <= 1.0


if __name__ == "__main__":
    test_round_trip_win_rate_basic()
    test_stop_loss_sell_is_counted()
    test_no_trades_returns_zero()
    test_orphan_sell_ignored()
    test_deprecated_win_rate_still_runs()
    print("test_win_rate: ALL PASSED")
