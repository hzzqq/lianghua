"""portfolio/advanced 打磨：覆盖各优化器合法性、权重清洗、基准守卫与批量对比。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.portfolio.advanced import (
    max_sharpe, min_cvar, shrinkage_min_var, max_entropy, momentum_score,
    min_tail_risk, vol_target_opt, bayesian_shrinkage, max_return,
    min_track_error, robust_cov, sanitize_weights, compare_optimizers,
)

COLS = ["A", "B", "C", "D"]


def _returns(n=300, seed=1):
    rng = np.random.default_rng(seed)
    # 4 个资产，弱相关、正收益
    base = rng.normal(0.0005, 0.01, size=(n, 4))
    base[:, 1] += 0.0003
    df = pd.DataFrame(base, columns=COLS)
    df.index = pd.date_range("2023-01-01", periods=n, freq="B")
    return df


def test_all_optimizers_valid_weights():
    r = _returns()
    for fn in (max_sharpe, min_cvar, shrinkage_min_var, max_entropy,
               momentum_score, min_tail_risk, vol_target_opt,
               bayesian_shrinkage, max_return, robust_cov):
        w = fn(r)
        assert isinstance(w, pd.Series)
        assert list(w.index) == COLS
        assert np.all(np.isfinite(w))
        assert w.min() >= -1e-9          # 多头约束
        assert abs(w.sum() - 1.0) < 1e-6  # 归一


def test_single_asset_degenerate():
    r = pd.DataFrame({"X": np.random.default_rng(2).normal(0.001, 0.01, 200)})
    for fn in (max_sharpe, shrinkage_min_var, max_entropy, min_cvar, robust_cov,
               bayesian_shrinkage):
        w = fn(r)
        assert abs(w.sum() - 1.0) < 1e-6
        assert w.iloc[0] == pytest.approx(1.0)


def test_nan_input_sanitized_to_finite():
    r = _returns()
    r.iloc[0, :] = np.nan
    r.iloc[5, 1] = np.nan
    for fn in (max_sharpe, shrinkage_min_var, bayesian_shrinkage, robust_cov):
        w = fn(r)
        assert np.all(np.isfinite(w))
        assert abs(w.sum() - 1.0) < 1e-6


def test_min_track_error_bench_length_guard():
    r = _returns()
    with pytest.raises(ValueError):
        min_track_error(r, bench=[0.5, 0.5])  # 长度 2 != 4
    # 正确长度 -> 完全贴合基准
    w = min_track_error(r, bench=[0.25, 0.25, 0.25, 0.25])
    assert np.allclose(w.values, 0.25)
    # 缺省等权
    w0 = min_track_error(r)
    assert np.allclose(w0.values, 0.25)


def test_sanitize_weights_long_only_and_degenerate():
    idx = ["x", "y", "z"]
    w = sanitize_weights(np.array([0.5, -0.2, 0.3]), idx, long_only=True)
    assert w.min() >= 0.0
    assert abs(w.sum() - 1.0) < 1e-9
    # 全 0 -> 等权退化
    w2 = sanitize_weights(np.array([0.0, 0.0, 0.0]), idx)
    assert np.allclose(w2, 1 / 3)
    # 非有限 -> 清零后归一
    w3 = sanitize_weights(np.array([1.0, np.nan, np.inf]), idx)
    assert np.all(np.isfinite(w3))
    assert abs(w3.sum() - 1.0) < 1e-9
    # 长度不一致 -> 报错
    with pytest.raises(ValueError):
        sanitize_weights(np.array([1.0, 2.0]), idx)


def test_compare_optimizers_shape():
    r = _returns()
    mat = compare_optimizers(r, names=["equal_weight", "max_sharpe", "min_variance"])
    assert mat.shape == (4, 3)
    assert list(mat.index) == COLS
    # 每列都是合法权重
    for c in mat.columns:
        assert abs(mat[c].sum() - 1.0) < 1e-6
