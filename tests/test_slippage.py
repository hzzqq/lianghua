"""slippage 冲击成本模型：守卫 + 新能力回归。"""
import math

import pytest

from lianghua.execution.slippage import (
    fill_price,
    round_trip_cost,
    slippage_cost,
    slippage_fraction,
)


def test_slippage_cost_basic():
    # 参与率 0.1%，冲击系数 0.1 -> 价格*0.1*0.001 = 0.01 金额成本
    assert abs(slippage_cost(100.0, 1000.0, 1_000_000.0, 0.1) - 0.01) < 1e-9


def test_slippage_cost_zero_adv_degrades():
    # 无流动性信息（adv<=0）退化为 0，不报错不崩
    assert slippage_cost(100.0, 1000.0, 0.0) == 0.0
    assert slippage_cost(100.0, 1000.0, -5.0) == 0.0


def test_slippage_cost_negative_impact_clamped():
    # impact_coef 误配为负（回扣）被钳为 0，避免负成本
    assert slippage_cost(100.0, 1000.0, 1_000_000.0, -0.5) == 0.0


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "abc"])
def test_slippage_cost_invalid_adv_raises(bad):
    with pytest.raises(ValueError):
        slippage_cost(100.0, 1000.0, bad, 0.1)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "x"])
def test_slippage_cost_invalid_price_qty_raises(bad):
    with pytest.raises(ValueError):
        slippage_cost(bad, 1000.0, 1_000_000.0)
    with pytest.raises(ValueError):
        slippage_cost(100.0, bad, 1_000_000.0)


def test_fill_price_buy_pays_more():
    p = fill_price(100.0, 1000.0, 1_000_000.0, spread=0.1, impact_coef=0.1)
    assert p > 100.0  # 买支付更高


def test_fill_price_sell_receives_less():
    p = fill_price(100.0, -1000.0, 1_000_000.0, spread=0.1, impact_coef=0.1)
    assert p < 100.0  # 卖收到更低


def test_fill_price_spread_direction():
    # 仅价差、无冲击：买卖价差的半 spread 对称
    buy = fill_price(100.0, 1.0, 1e9, spread=0.2, impact_coef=0.0)
    sell = fill_price(100.0, -1.0, 1e9, spread=0.2, impact_coef=0.0)
    assert abs((buy - 100.0) - (100.0 - sell)) < 1e-9
    assert abs(buy - sell - 0.2) < 1e-9


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_fill_price_invalid_raises(bad):
    with pytest.raises(ValueError):
        fill_price(bad, 1000.0, 1_000_000.0)
    with pytest.raises(ValueError):
        fill_price(100.0, bad, 1_000_000.0)


def test_slippage_fraction_range():
    frac = slippage_fraction(100.0, 1000.0, 1_000_000.0, spread=0.1, impact_coef=0.1)
    assert 0.0 < frac < 1.0


def test_slippage_fraction_zero_price():
    assert slippage_fraction(0.0, 1000.0, 1_000_000.0) == 0.0


def test_round_trip_cost_symmetric():
    rt = round_trip_cost(100.0, 1000.0, 1_000_000.0, spread=0.1, impact_coef=0.1)
    # 往返 = 价差全额 + 双边冲击
    assert rt > 0.0
    single = abs(fill_price(100.0, 1000.0, 1_000_000.0, 0.1, 0.1) - 100.0)
    assert rt >= 2 * single - 1e-9


def test_round_trip_cost_zero_adv():
    assert round_trip_cost(100.0, 1000.0, 0.0, spread=0.1) == pytest.approx(0.1)
