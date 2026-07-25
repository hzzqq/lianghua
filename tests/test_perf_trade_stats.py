"""perf.trade_stats 测试：数组输入、连胜连败、盈亏比、NaN 守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.perf.trade_stats import trade_stats


def test_basic_and_runs():
    pnl = [0.10, -0.05, 0.20, 0.15, -0.10, -0.20, 0.05]  # 连胜2, 连败2
    s = trade_stats(pnl)
    assert s["num_trades"] == 7
    assert s["win_rate"] == pytest.approx(4 / 7)
    assert s["max_consecutive_wins"] == 2
    assert s["max_consecutive_losses"] == 2
    assert s["payoff_ratio"] > 1.0  # 盈利均值/亏损均值
    assert s["expectancy"] == pytest.approx(np.mean(pnl))


def test_empty():
    s = trade_stats([])
    assert s["num_trades"] == 0
    assert s["win_rate"] == 0.0


def test_array_and_series_input():
    arr = np.array([0.1, -0.2, 0.3])
    s_arr = trade_stats(arr)
    s_ser = trade_stats(pd.Series(arr))
    assert s_arr["num_trades"] == 3
    assert s_ser["num_trades"] == 3
    assert s_arr["expectancy"] == pytest.approx(s_ser["expectancy"])


def test_dict_input_none_compat():
    # pnl=None 回落为 0（兼容历史调用），不得抛错
    s = trade_stats([{"side": "BUY", "pnl": None}, {"pnl": 0.1}, {"pnl": -0.05}])
    assert s["num_trades"] == 3


def test_nan_raises():
    # 隐性修复验证：NaN pnl 必须显式报错而非静默清零
    with pytest.raises(ValueError):
        trade_stats([0.1, np.nan, -0.05])
    with pytest.raises(ValueError):
        trade_stats(pd.Series([0.1, np.inf, -0.05]))


def test_all_wins_profit_factor_inf():
    s = trade_stats([0.1, 0.2, 0.3])
    assert s["profit_factor"] == float("inf")
    assert s["payoff_ratio"] == float("inf")
    assert s["max_consecutive_wins"] == 3
    assert s["max_consecutive_losses"] == 0


def test_break_even_excluded_from_win_loss():
    # pnl=0 既不计入胜也不计入负，连败归零
    s = trade_stats([0.0, 0.0, 0.1])
    assert s["win_rate"] == pytest.approx(1 / 3)
    assert s["max_consecutive_losses"] == 0
