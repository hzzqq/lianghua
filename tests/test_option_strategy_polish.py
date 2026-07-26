"""迭代 44：期权组合损益分析 + Leg 入参校验打磨。"""
import math
import numpy as np
import pandas as pd

from lianghua.option.strategy import (
    Leg, combo_payoff, payoff_curve, analyze_combo,
    iron_condor, butterfly, straddle, covered_call, get_option_combo,
)


def test_leg_validation():
    # 无效 side（"BUY" lower 后为合法 buy，故不在此列）
    for bad in ["hold", 1, "", "SELLX"]:
        try:
            Leg(bad, "CALL", 100, 1)
            raise AssertionError("应拒绝无效 side")
        except ValueError:
            pass
    # 无效 otype
    try:
        Leg("buy", "FOO", 100, 1)
        raise AssertionError("应拒绝无效 otype")
    except ValueError:
        pass
    # 负/非有限权利金
    for p in [-1.0, math.nan, math.inf]:
        try:
            Leg("buy", "CALL", 100, p)
            raise AssertionError("应拒绝非法权利金")
        except ValueError:
            pass
    # 负行权价
    try:
        Leg("buy", "CALL", -5, 1)
        raise AssertionError("应拒绝负行权价")
    except ValueError:
        pass
    # stock 自动归一 otype
    assert Leg("buy", "CALL", 0, 0, kind="stock").otype == "STOCK"


def test_combo_payoff_empty_raises():
    try:
        combo_payoff([], 100)
        raise AssertionError("空 legs 应抛错")
    except ValueError:
        pass
    try:
        payoff_curve([], 90, 110)
        raise AssertionError("空 legs 应抛错")
    except ValueError:
        pass


def test_payoff_curve_bounds():
    # S_hi <= S_lo 应抛错
    try:
        payoff_curve([Leg("buy", "CALL", 100, 3)], 110, 90)
        raise AssertionError("S_hi<=S_lo 应抛错")
    except ValueError:
        pass
    # 非有限边界
    try:
        payoff_curve([Leg("buy", "CALL", 100, 3)], math.nan, 110)
        raise AssertionError("非有限边界应抛错")
    except ValueError:
        pass


def test_analyze_long_call():
    legs = get_option_combo("long_call", strike=100, premium=3)
    a = analyze_combo(legs, 80, 120)
    # 买 CALL：最大亏损=权利金(-3)；区间内(80~120)盈利有限=17（非 inf，因区间有界）
    assert a["max_loss"] == -3.0
    assert a["max_profit"] > 0 and math.isfinite(a["max_profit"])
    assert a["max_profit"] == 17.0
    # 盈亏平衡 ≈ 103
    assert len(a["breakevens"]) == 1
    assert abs(a["breakevens"][0] - 103.0) < 0.5
    assert a["net_premium"] == -3.0


def test_analyze_iron_condor():
    legs = get_option_combo("iron_condor")
    a = analyze_combo(legs, 80, 120)
    # 铁鹰：最大盈利有限，最大亏损有限
    assert math.isfinite(a["max_profit"])
    assert math.isfinite(a["max_loss"])
    assert a["max_loss"] < 0 < a["max_profit"]
    # 应有 2 个盈亏平衡点
    assert len(a["breakevens"]) == 2


def test_analyze_butterfly():
    legs = get_option_combo("butterfly")
    a = analyze_combo(legs, 80, 120)
    # 该默认蝶式为净收权利金(净贷方)，故全区间盈利、无盈亏平衡点
    assert a["max_loss"] > 0
    assert a["max_profit"] > a["max_loss"]
    # 净贷方蝶式无穿越 0 的点
    assert len(a["breakevens"]) == 0


def test_analyze_covered_call():
    legs = get_option_combo("covered_call")
    a = analyze_combo(legs, 80, 120)
    # 备兑看涨：上行封顶，最大亏损有限（股票-权利金）
    assert math.isfinite(a["max_loss"])
    assert a["max_profit"] > 0
    assert len(a["breakevens"]) == 1


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print("OK", fn.__name__)
    print("ALL option strategy polish tests passed")
