"""backtest/orchestrator 单元测试：净值归一化守卫 + walk_forward_weights 新能力。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.backtest.orchestrator import (
    _normalize_to_base,
    walk_forward_weights,
)


def _returns(n=60, m=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = [f"A{i}" for i in range(m)]
    return pd.DataFrame(rng.normal(0.001, 0.01, (n, m)), index=idx, columns=cols)


def test_normalize_base_safe_with_zero_first():
    # 隐性修复：首个净值为 0 不应产出 inf
    eq = pd.Series([0.0, 1.0, 2.0, 1.5])
    norm = _normalize_to_base(eq)
    assert np.all(np.isfinite(norm.values))
    assert not np.isinf(norm).any()


def test_normalize_base_keeps_unit_start():
    eq = pd.Series([100.0, 110.0, 90.0])
    norm = _normalize_to_base(eq)
    assert abs(norm.iloc[0] - 1.0) < 1e-12


def test_walk_forward_weights_shape_and_sum():
    r = _returns()
    eq_w = lambda rets: pd.Series(1.0 / rets.shape[1], index=rets.columns)
    wf = walk_forward_weights(r, eq_w, rebalance="ME")
    assert wf.shape[1] == r.shape[1]
    assert wf.shape[0] >= 1
    assert np.all(np.isfinite(wf.values))
    # 每行权重和应为 1
    assert np.allclose(wf.sum(axis=1).values, 1.0, atol=1e-9)


def test_walk_forward_weights_invalid_freq():
    r = _returns()
    eq_w = lambda rets: pd.Series(1.0 / rets.shape[1], index=rets.columns)
    with pytest.raises(ValueError):
        walk_forward_weights(r, eq_w, rebalance="NOTAFREQ")


def test_walk_forward_weights_rejects_bad_input():
    eq_w = lambda rets: pd.Series(1.0 / rets.shape[1], index=rets.columns)
    with pytest.raises(TypeError):
        walk_forward_weights([1, 2, 3], eq_w)
    with pytest.raises(ValueError):
        walk_forward_weights(pd.DataFrame(), eq_w)
