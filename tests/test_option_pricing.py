"""期权定价模块测试（Round 15）：新增美式二叉树定价 + 隐含波动率边界/输入守卫。"""
import math
import numpy as np
import pytest

from lianghua.option.pricing import bs_price, greeks, implied_vol, binomial_price


def test_bs_put_call_parity():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    c = bs_price(S, K, T, r, sigma, "CALL")
    p = bs_price(S, K, T, r, sigma, "PUT")
    # C - P = S - K e^{-rT}
    assert abs((c - p) - (S - K * math.exp(-r * T))) < 1e-6


def test_invalid_opt_type_raises():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0.2, "FOO")
    with pytest.raises(ValueError):
        greeks(100, 100, 1, 0.05, 0.2, "BAR")
    with pytest.raises(ValueError):
        implied_vol(5, 100, 100, 1, 0.05, "BAD")
    with pytest.raises(ValueError):
        binomial_price(100, 100, 1, 0.05, 0.2, "XYZ")


def test_implied_vol_recovers_sigma():
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.30
    price = bs_price(S, K, T, r, sigma, "CALL")
    iv = implied_vol(price, S, K, T, r, "CALL")
    assert abs(iv - sigma) < 1e-3


def test_implied_vol_outside_no_arb_bounds_returns_nan():
    # 目标价远超无套利区间 -> 无解
    iv = implied_vol(999.0, 100, 100, 1, 0.05, "CALL")
    assert math.isnan(iv)


def test_implied_vol_below_intrinsic_returns_zero():
    # 价格 <= 内在价值: 无正波动率解
    iv = implied_vol(10.0, 120, 100, 1, 0.05, "CALL")  # 内在价值=20, 10<=20
    assert iv == 0.0


def test_implied_vol_bad_inputs_nan():
    assert math.isnan(implied_vol(5, -100, 100, 1, 0.05, "CALL"))
    assert math.isnan(implied_vol(float("nan"), 100, 100, 1, 0.05, "CALL"))


def test_binomial_matches_bs_for_european():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    bs = bs_price(S, K, T, r, sigma, "CALL")
    bino = binomial_price(S, K, T, r, sigma, "CALL", steps=400, american=False)
    assert abs(bs - bino) < 0.1


def test_binomial_american_ge_european():
    S, K, T, r, sigma = 100.0, 90.0, 1.0, 0.05, 0.4
    euro = binomial_price(S, K, T, r, sigma, "CALL", steps=200, american=False)
    amer = binomial_price(S, K, T, r, sigma, "CALL", steps=200, american=True)
    # 美式看涨在無股息时不应低于欧式，且深实值美式可提前行权获利
    assert amer >= euro - 1e-9


def test_binomial_degenerate_returns_intrinsic():
    # T<=0 退化
    assert binomial_price(120, 100, 0, 0.05, 0.2, "CALL") == 20.0
    # steps<=0 退化兜底
    assert binomial_price(120, 100, 1, 0.05, 0.2, "CALL", steps=0) == 20.0


def test_greeks_put_call_sanity():
    g = greeks(100, 100, 1, 0.05, 0.2, "CALL")
    assert 0.0 < g["delta"] < 1.0
    assert g["gamma"] > 0
    assert g["vega"] > 0
    gp = greeks(100, 100, 1, 0.05, 0.2, "PUT")
    assert -1.0 < gp["delta"] < 0.0
    # put-call parity on delta: delta_call - delta_put = 1
    assert abs(g["delta"] - gp["delta"] - 1.0) < 1e-9
