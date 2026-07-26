"""Cycle 51：portfolio.target_vol 打磨——inf 泄漏/参数守卫/杠杆上限 + vol_target_scalar。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.portfolio.target_vol import target_vol_weights, vol_target_scalar


def _rets(seed=0, n=250, cols=("A", "B", "C"), scale=0.01):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, scale, (n, len(cols))), columns=cols)


def test_inf_does_not_produce_nan_weights():
    # 隐性修复：inf 收益不应静默产出 NaN/0 权重
    df = _rets()
    df.loc[df.index[0], "A"] = np.inf
    df.loc[df.index[1], "B"] = -np.inf
    w = target_vol_weights(df)
    assert np.all(np.isfinite(w.values))
    # 缩放因子有限
    s = vol_target_scalar(df)
    assert np.isfinite(s)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        target_vol_weights(_rets(), target_vol=0.0)
    with pytest.raises(ValueError):
        target_vol_weights(_rets(), periods=0)
    with pytest.raises(ValueError):
        target_vol_weights(_rets(), max_leverage=-1)
    with pytest.raises(TypeError):
        target_vol_weights([1, 2, 3])


def test_max_leverage_caps_scalar():
    # 极低波动 -> 无上限时 scale 很大；上限应夹紧
    tiny = _rets(scale=1e-5)  # 极小的已实现波动
    s_free = vol_target_scalar(tiny)
    assert s_free > 5  # 杠杆明显放大
    s_capped = vol_target_scalar(tiny, max_leverage=2.0)
    assert s_capped == 2.0
    w = target_vol_weights(tiny, max_leverage=2.0)
    assert np.allclose(w.max(), 1.0 / 3 * 2.0, atol=1e-9)


def test_weights_equal_w0_times_scalar():
    df = _rets(seed=4)
    w = target_vol_weights(df, target_vol=0.10)
    s = vol_target_scalar(df, target_vol=0.10)
    assert np.allclose(w.values, np.full(3, 1.0 / 3) * s, atol=1e-9)


def test_constant_returns_falls_back_to_equal():
    const = pd.DataFrame({"A": [0.001] * 100, "B": [0.001] * 100, "C": [0.001] * 100})
    w = target_vol_weights(const)
    assert np.allclose(w.values, 1.0 / 3, atol=1e-9)
