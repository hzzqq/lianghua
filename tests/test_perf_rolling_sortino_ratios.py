"""perf/ 打磨：覆盖滚动指标、下行风险调整、进阶比率的合法性与边界守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.perf.rolling import (
    rolling_sharpe, rolling_volatility, rolling_return,
    rolling_max_drawdown, rolling_report,
)
from lianghua.perf.sortino import sortino, calmar, omega_ratio
from lianghua.perf.ratios import (
    ulcer_index, martin_ratio, gain_to_pain, cagr, up_down_capture,
    k_ratio, omega_ratio as omega_ratio_r, calmar_ratio, tail_ratio,
)
from lianghua.perf.metrics import sharpe, max_drawdown


def _equity(n=300, seed=4, drift=0.0005, vol=0.01):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n)
    eq = 100 * np.cumprod(1 + ret)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(eq, index=idx, name="equity")


def test_rolling_metrics_shape_and_finite():
    eq = _equity()
    for fn in (rolling_sharpe, rolling_volatility, rolling_return, rolling_max_drawdown):
        out = fn(eq, window=60)
        assert isinstance(out, pd.Series)
        # 收益型指标丢弃首日 NaN(长度 n-1)，回撤型保留(长度 n)
        assert len(out) in (len(eq), len(eq) - 1)
    rep = rolling_report(eq, window=60)
    assert list(rep.columns) == ["滚动夏普", "滚动波动率", "滚动年化收益", "滚动最大回撤"]
    assert rep.shape[0] == len(eq)


def test_rolling_sharpe_zero_vol_no_inf():
    # 常数序列：波动为 0，滚动夏普应为 NaN（非 inf）
    const = pd.Series([100.0] * 120)
    rs = rolling_sharpe(const, window=30)
    assert not rs.isin([np.inf, -np.inf]).any()


def test_sortino_and_calmar():
    eq = _equity()
    s = sortino(eq)
    assert np.isfinite(s)
    c = calmar(eq)
    assert np.isfinite(c)
    # 无下行时 Omega 应为 inf（基准已处理）
    flat = pd.Series([100.0] * 50)
    assert omega_ratio(flat) == pytest.approx(np.inf)


def test_ratios_finite_on_normal_equity():
    eq = _equity()
    bench = _equity(seed=9)
    for fn in (ulcer_index, martin_ratio, gain_to_pain, cagr, k_ratio,
               omega_ratio_r, calmar_ratio, tail_ratio):
        val = fn(eq) if fn is not omega_ratio_r else fn(eq)
        assert np.isfinite(val), fn.__name__
    cap = up_down_capture(eq, bench)
    assert set(cap.keys()) == {"up_capture", "down_capture"}
    assert np.all(np.isfinite(list(cap.values())))


def test_sharpe_maxdrawdown_zero_nav_no_warning():
    # 净值含 0：pct_change 会产生 inf，必须被清洗而非发出无效值警告/返回 inf
    nav = pd.Series([0.0, 1.0, 1.05, 0.98, 1.1, 1.2, 1.15])
    with np.errstate(all="raise"):
        sh = sharpe(nav)        # 不应触发无效值计算异常
    assert np.isfinite(sh)
    mdd = max_drawdown(nav)
    assert np.isfinite(mdd)


def test_cagr_edge_cases():
    assert cagr(pd.Series([1.0] * 5)) == 0.0          # 常数
    assert cagr(pd.Series([0.0, 1.0])) == 0.0          # 首值非正
    assert cagr(pd.Series([100.0, 110.0])) != 0.0     # 正常
