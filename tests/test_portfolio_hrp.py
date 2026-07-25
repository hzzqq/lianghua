"""HRP 层次风险平价测试（Round 17）：新增 returns API + 隐性守卫（零方差/奇异/非方阵/NaN）。"""
import numpy as np
import pytest

from lianghua.portfolio.hrp import hrp, hrp_from_returns


def test_hrp_basic_weights_sum_one():
    rng = np.random.default_rng(0)
    cov = rng.random((5, 5))
    cov = cov @ cov.T + np.eye(5) * 0.1  # 正定
    w = hrp(cov)
    assert np.all(w >= 0)
    assert abs(w.sum() - 1.0) < 1e-9


def test_hrp_zero_variance_assets_no_nan():
    # 部分资产零方差 -> 旧实现 rl+rr=0 产生 NaN
    cov = np.diag([0.0, 0.0, 0.5, 0.2]).astype(float)
    w = hrp(cov)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-9


def test_hrp_non_square_raises():
    with pytest.raises(ValueError):
        hrp(np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3]]))


def test_hrp_nan_cov_raises():
    cov = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.raises(ValueError):
        hrp(cov)


def test_hrp_singular_from_collinear_returns():
    # 两资产完全共线 -> 样本协方差奇异，但直接 hrp(cov) 收到原始奇异矩阵会 NaN
    rets = np.array([[0.01, 0.01], [0.02, 0.02], [-0.01, -0.01], [0.03, 0.03]])
    cov = np.cov(rets.T)
    w = hrp(cov)  # 原始奇异矩阵：应安全（等权兜底）而非 NaN
    assert np.all(np.isfinite(w))


def test_hrp_from_returns_shrinkage_stable():
    rng = np.random.default_rng(1)
    # 高共线收益
    base = rng.normal(0, 1, 50)
    rets = np.column_stack([base, base * 0.99, base * 1.01, rng.normal(0, 0.1, 50)])
    w = hrp_from_returns(rets, shrink=0.2)
    assert np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w >= 0)


def test_hrp_from_returns_insufficient_data():
    rets = np.array([[0.01, 0.02]])
    w = hrp_from_returns(rets)
    assert np.all(np.isfinite(w)) and abs(w.sum() - 1.0) < 1e-9


def test_hrp_from_returns_bad_shape():
    with pytest.raises(ValueError):
        hrp_from_returns(np.array([1.0, 2.0, 3.0]))
