"""参数优化：新增 random_search/best_params + 空网格/失败隔离/守卫回归测试。

覆盖：
- 空 param_grid 不再 KeyError（隐性崩溃）
- 单组参数异常不中止整轮搜索（失败隔离）
- run_backtest=None 时构造买入持有 proxy（与文档一致，不再静默 NaN）
- random_search 可复现 + best_params 取最优
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.optimize.param_search import (
    grid_search,
    random_search,
    walk_forward,
    best_params,
)


def _df(n=400):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.cumprod(1.0 + ret)
    return pd.DataFrame({"close": close}, index=idx)


def _signals(df, window=10, **_kw):
    return pd.Series(1, index=df.index)


def _rb(df, sig):
    ret = df["close"].pct_change().fillna(0.0)
    eq = (1.0 + sig.reindex(df.index).fillna(0.0) * ret).cumprod() * 100.0
    return eq


def test_grid_search_returns_sorted():
    df = _df()
    res = grid_search(df, _signals, {"window": [5, 10, 20]}, run_backtest=_rb)
    assert "score" in res.columns
    assert res["score"].is_monotonic_decreasing
    assert len(res) == 3


def test_empty_param_grid_raises():
    df = _df()
    with pytest.raises(ValueError):
        grid_search(df, _signals, {}, run_backtest=_rb)


def test_per_combo_failure_isolation():
    def flaky(df, window=10):
        if window == 0:
            raise ValueError("bad window")
        return pd.Series(1, index=df.index)

    df = _df()
    res = grid_search(df, flaky, {"window": [0, 5, 10]}, run_backtest=_rb)
    # 2 组成功、1 组记录错误而非整轮崩溃
    assert res["score"].notna().sum() == 2
    bad = res.loc[res["window"] == 0]
    assert (bad["error"].str.len() > 0).all()


def test_proxy_when_no_run_backtest():
    df = _df()
    res = grid_search(df, _signals, {"window": [5, 10]}, run_backtest=None)
    assert res["score"].notna().all()


def test_random_search_runs_and_reproducible():
    df = _df()
    dist = {"window": [5, 10, 20, 30], "k": (1, 4, "int")}
    a = random_search(df, _signals, dist, n_iter=30, run_backtest=_rb, seed=42)
    b = random_search(df, _signals, dist, n_iter=30, run_backtest=_rb, seed=42)
    assert len(a) == 30
    assert a["score"].is_monotonic_decreasing
    pd.testing.assert_frame_equal(a, b)


def test_random_search_invalid_dist_raises():
    df = _df()
    with pytest.raises(ValueError):
        random_search(df, _signals, {"window": 5}, n_iter=5, run_backtest=_rb)


def test_random_search_invalid_n_raises():
    df = _df()
    with pytest.raises(ValueError):
        random_search(df, _signals, {"window": [5]}, n_iter=0, run_backtest=_rb)


def test_walk_forward_returns_oos():
    df = _df(400)
    oos, best = walk_forward(
        df, _signals, {"window": [5, 10, 20]},
        train_size=250, test_size=60, run_backtest=_rb,
    )
    assert len(oos) >= 1
    assert all(np.isfinite(oos))
    assert len(best) == len(oos)


def test_walk_forward_guards():
    df = _df(400)
    with pytest.raises(ValueError):
        walk_forward(df, _signals, {"window": [5]}, train_size=0, test_size=60)
    with pytest.raises(ValueError):
        walk_forward(df, _signals, {"window": [5]}, train_size=250, test_size=-1)


def test_best_params_picks_top():
    df = _df()
    res = grid_search(df, _signals, {"window": [5, 10, 20]}, run_backtest=_rb)
    bp = best_params(res)
    assert isinstance(bp, dict)
    assert bp["window"] in (5, 10, 20)


def test_best_params_empty():
    assert best_params(pd.DataFrame()) is None
    assert best_params(None) is None
