"""迭代 48：risk/liquidity 打磨 —— 输入校验 + amihud_series + 权重感知冲击成本。"""
import numpy as np
import pandas as pd

from lianghua.risk.liquidity import amihud, amihud_series, liquidity_cost


def _series(*vals):
    return pd.Series(vals, dtype=float)


def test_amihud_basic():
    # ret 1%, price 100, vol 10000 -> |0.01|/(100*10000)=1e-8
    r = _series(0.01, 0.02, -0.01)
    v = _series(10000.0, 10000.0, 10000.0)
    p = _series(100.0, 100.0, 100.0)
    val = amihud(r, v, p)
    # 均值 = (1e-8 + 2e-8 + 1e-8) / 3 = 1.333e-8
    assert abs(val - 4e-8 / 3) < 1e-12


def test_amihud_series_alignment():
    r = _series(0.01, 0.0, -0.02, 0.0)
    v = _series(10000.0, 10000.0, 10000.0, 10000.0)
    p = _series(100.0, 100.0, 100.0, 100.0)
    s = amihud_series(r, v, p)
    assert len(s) == 4
    # 零收益 -> 0 贡献；零成交额 -> NaN
    assert s.iloc[1] == 0.0


def test_amihud_zero_volume_raises():
    import pytest
    r = _series(0.01, 0.02)
    v = _series(0.0, 0.0)
    p = _series(100.0, 100.0)
    with pytest.raises(ValueError):
        amihud(r, v, p)


def test_amihud_invalid_input():
    import pytest
    # 非 Series
    with pytest.raises(TypeError):
        amihud([0.01, 0.02], _series(1.0), _series(1.0))
    # 负成交量
    with pytest.raises(ValueError):
        amihud(_series(0.01), _series(-1.0), _series(1.0))
    # 全 NaN 对齐后为空
    with pytest.raises(ValueError):
        amihud(_series(0.01, np.nan), _series(0.0, 1.0), _series(1.0, 1.0))


def test_liquidity_cost_weight_aware():
    weights = {"A": 0.5, "B": 0.5}
    adv = {"A": 1e6, "B": 1e6}
    df = liquidity_cost(weights, adv, participant_rate=0.1, impact_k=0.1)
    # cost% = 0.1*sqrt(0.1)*0.5 = 0.1*0.3162*0.5 ≈ 1.58%
    assert abs(df.loc[df["资产"] == "A", "冲击成本%"].iloc[0] - 0.1 * np.sqrt(0.1) * 0.5 * 100) < 1e-6
    # 权重更大 -> 成本更高
    df2 = liquidity_cost({"A": 1.0}, adv, participant_rate=0.1, impact_k=0.1)
    assert df2["冲击成本%"].iloc[0] > df["冲击成本%"].iloc[0]


def test_liquidity_cost_unknown_adv_worst_case():
    weights = {"A": 0.5}
    # 未知 ADV -> 按最坏参与度(part=1.0)估计，高于已知 ADV 情形
    df_unknown = liquidity_cost(weights, {}, participant_rate=0.1, impact_k=0.1)
    df_known = liquidity_cost(weights, {"A": 1e6}, participant_rate=0.1, impact_k=0.1)
    assert df_unknown["冲击成本%"].iloc[0] > df_known["冲击成本%"].iloc[0]


def test_liquidity_cost_validation():
    import pytest
    with pytest.raises(ValueError):
        liquidity_cost({"A": 0.5}, {"A": 1e6}, participant_rate=1.5)
