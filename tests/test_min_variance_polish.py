"""R39: min_variance 数值稳定化验收。"""
import numpy as np
import pandas as pd

from lianghua.portfolio.min_variance import (
    min_variance, min_variance_from_returns,
)


def test_min_variance_basic():
    cov = np.array([[0.04, 0.006], [0.006, 0.09]])
    w = min_variance(cov)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_min_variance_singular_uses_pinv():
    # 完全共线 -> 奇异协方差，裸 inv 会崩；pinv 应稳定返回等权倾向
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])
    w = min_variance(cov, ridge=0.1)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0)


def test_min_variance_nonfinite_raises():
    cov = np.array([[1.0, np.nan], [np.nan, 1.0]])
    try:
        min_variance(cov)
    except ValueError:
        return
    raise AssertionError("非有限 cov 应抛 ValueError")


def test_min_variance_non_square_raises():
    try:
        min_variance(np.array([[1.0, 0.0, 0.0]]))
    except ValueError:
        return
    raise AssertionError("非方阵应抛 ValueError")


def test_min_variance_single_asset():
    w = min_variance(np.array([[0.05]]))
    assert np.allclose(w, [1.0])


def test_min_variance_ridge_pulls_toward_equal():
    # 强 ridge 时权重应更接近等权
    cov = np.array([[0.01, 0.0], [0.0, 1.0]])
    w = min_variance(cov, ridge=10.0)
    assert np.allclose(w, [0.5, 0.5], atol=0.05)


def test_from_returns_end_to_end():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0.0005, 0.01, (120, 3)),
                     columns=["a", "b", "c"])
    w = min_variance_from_returns(r)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_from_returns_collinear_stable():
    # 两列完全共线 -> 样本协方差奇异，应稳定
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, 100)
    r = pd.DataFrame({"x": base, "y": base})
    w = min_variance_from_returns(r, ridge=0.2)
    assert np.all(np.isfinite(w)) and np.isclose(w.sum(), 1.0)


def test_from_returns_empty_or_single():
    assert np.allclose(min_variance_from_returns(
        pd.DataFrame(columns=["a"])), [1.0])
    assert min_variance_from_returns(pd.DataFrame()).size == 0
