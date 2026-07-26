"""迭代 46：risk/position_sizing 打磨 —— 入参守卫 + 分数凯利 + kelly_from_returns。"""
import numpy as np
import pandas as pd

from lianghua.risk.position_sizing import (
    kelly_fraction, fixed_fractional, vol_target_size, kelly_from_returns,
)


def test_kelly_fraction_basic():
    # 60% 胜率，赔率 1:1 -> f = 0.6 - 0.4 = 0.2
    assert abs(kelly_fraction(0.6, 1.0) - 0.2) < 1e-9
    # 赔率<=0 不下注
    assert kelly_fraction(0.9, 0.0) == 0.0
    # 分数凯利（半凯利）
    assert abs(kelly_fraction(0.6, 1.0, fraction=0.5) - 0.1) < 1e-9


def test_kelly_fraction_validation():
    import pytest
    # win_rate 越界显式抛错
    for bad in [-0.1, 1.5, "x"]:
        with pytest.raises(ValueError):
            kelly_fraction(bad, 1.0)
    # fraction 越界
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 1.0, fraction=1.5)


def test_fixed_fractional():
    # 10000 * 0.02 / 5 = 40
    assert abs(fixed_fractional(10000, 0.02, 5) - 40.0) < 1e-9
    # 非法输入返回 0（不产出负/inf 仓位）
    assert fixed_fractional(-100, 0.02, 5) == 0.0
    assert fixed_fractional(10000, 1.5, 5) == 0.0
    assert fixed_fractional(10000, 0.02, 0) == 0.0
    import math
    assert fixed_fractional(10000, math.nan, 5) == 0.0


def test_vol_target_size():
    # 目标日波动 = 0.01*10000 = 100；每股日波动 = 10*0.02 = 0.2 -> 500 股
    assert abs(vol_target_size(10000, 10, 0.02, target_vol=0.01) - 500.0) < 1e-9
    # 退化输入返回 0
    assert vol_target_size(0, 10, 0.02) == 0.0
    assert vol_target_size(10000, 10, 0.0) == 0.0
    import math
    assert vol_target_size(10000, math.inf, 0.02) == 0.0


def test_kelly_from_returns():
    # 小凯利序列（mu/var < 1，避免被 [0,1] 截断，便于验证分数关系）
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.01, 0.2, 500))
    f = kelly_from_returns(rets)
    assert 0.0 <= f <= 1.0
    # 半凯利应为全凯利的一半
    f_half = kelly_from_returns(rets, fraction=0.5)
    assert abs(f_half - f * 0.5) < 1e-9
    # 分数为 0.25 时结果与 full*0.25 一致
    assert abs(kelly_from_returns(rets, fraction=0.25) - f * 0.25) < 1e-9
    # 方差为 0（恒定收益）不下注
    assert kelly_from_returns(pd.Series([0.01, 0.01, 0.01])) == 0.0
    # 不足观测返回 0
    assert kelly_from_returns(pd.Series([0.01])) == 0.0
    # 含 NaN 被过滤
    s = pd.Series([0.02, np.nan, 0.03, np.inf])
    assert 0.0 <= kelly_from_returns(s) <= 1.0
