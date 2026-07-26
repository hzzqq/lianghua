"""R40: black_litterman 健壮性验收。"""
import numpy as np

from lianghua.portfolio.black_litterman import (
    black_litterman, black_litterman_weights,
)


def _setup(n=3, k=2, seed=0):
    rng = np.random.default_rng(seed)
    pi = rng.normal(0.05, 0.02, n)
    A = rng.normal(0, 0.1, (n, n))
    cov = A @ A.T / n + np.eye(n) * 0.01
    P = np.eye(k, n)  # 前 k 个资产各一个绝对观点
    Q = rng.normal(0.06, 0.01, k)
    return pi, cov, P, Q


def test_basic_posterior_finite():
    pi, cov, P, Q = _setup()
    post = black_litterman(pi, cov, P, Q)
    assert np.all(np.isfinite(post))
    assert post.shape == pi.shape


def test_singular_cov_stable_with_pinv():
    # cov 奇异（秩亏）：裸 inv 会崩，pinv 应稳定
    pi, _, P, Q = _setup()
    cov = np.ones((3, 3)) * 0.04  # 秩 1
    post = black_litterman(pi, cov, P, Q, ridge=1e-6)
    assert np.all(np.isfinite(post))


def test_shape_mismatch_raises():
    pi, cov, P, Q = _setup()
    try:
        black_litterman(pi, cov, P, Q[:-1])  # Q 长度与 P 行数不符
    except ValueError:
        return
    raise AssertionError("Q/P 形状不符应抛 ValueError")


def test_cov_not_square_raises():
    pi, _, P, Q = _setup()
    try:
        black_litterman(pi, np.array([[1.0, 0.0, 0.0]]), P, Q)
    except ValueError:
        return
    raise AssertionError("非方阵 cov 应抛 ValueError")


def test_nonfinite_raises():
    pi, cov, P, Q = _setup()
    pi[0] = np.nan
    try:
        black_litterman(pi, cov, P, Q)
    except ValueError:
        return
    raise AssertionError("非有限输入应抛 ValueError")


def test_weights_end_to_end_normalized():
    pi, cov, P, Q = _setup()
    w = black_litterman_weights(pi, cov, P, Q)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_weights_always_valid_and_normalized():
    # 任意（含极端）后验输入都应产出归一化、非负、有限的权重
    pi, cov, P, Q = _setup()
    for mult in (1.0, -1.0, 100.0, -100.0):
        w = black_litterman_weights(mult * pi, cov, P, Q)
        assert np.all(np.isfinite(w))
        assert np.isclose(w.sum(), 1.0)
        assert (w >= 0).all()


def test_omega_shape_validated():
    pi, cov, P, Q = _setup()
    try:
        black_litterman(pi, cov, P, Q, omega=np.eye(3))  # 应为 k×k
    except ValueError:
        return
    raise AssertionError("omega 形状错误应抛 ValueError")
