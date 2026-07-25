"""risk/tail 单元测试：权重对齐修复 + diversification_ratio 新能力。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.risk.tail import (
    component_var,
    diversification_ratio,
    risk_contribution,
)


def _rets(n=120, m=3, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = [f"A{i}" for i in range(m)]
    return pd.DataFrame(rng.normal(0.001, 0.01, (n, m)), index=idx, columns=cols)


def test_component_var_aligns_series_order():
    # 隐性修复：权重 Series 顺序与列不一致时原实现会错位
    r = _rets()
    cols = list(r.columns)
    w_series = pd.Series([0.2, 0.5, 0.3], index=cols)           # 正确顺序
    w_scrambled = w_series.reindex([cols[2], cols[0], cols[1]])  # 打乱顺序
    a = component_var(r, w_series)
    b = component_var(r, w_scrambled)
    assert np.allclose(a.values, b.values, atol=1e-9)
    assert list(a.index) == cols


def test_risk_contribution_aligns_dict():
    r = _rets()
    cols = list(r.columns)
    w_dict = {cols[0]: 0.6, cols[1]: 0.3, cols[2]: 0.1}
    out = risk_contribution(r, w_dict)
    assert list(out.index) == cols
    assert np.all(np.isfinite(out.values))


def test_align_weights_length_mismatch_raises():
    r = _rets()
    with pytest.raises(ValueError):
        component_var(r, [0.5, 0.5])  # 长度 2 != 3


def test_diversification_ratio_uncorrelated_gt_one():
    r = _rets(seed=5)
    # 等权 + 低相关：分散化比率应 > 1
    dr = diversification_ratio(r, [1 / 3, 1 / 3, 1 / 3])
    assert dr > 1.0
    assert np.isfinite(dr)


def test_diversification_ratio_single_asset_is_one():
    r = _rets()
    dr = diversification_ratio(r, [1.0, 0.0, 0.0])
    assert abs(dr - 1.0) < 1e-9
