"""迭代 141–150（第九轮）集成测试。

覆盖：
- 5 个新策略（经统一注册表）输出 {-1,0,1}、长度对齐
- 4 个新优化器（经统一注册表）输出权重和≈1、非负
- 1 个新因子函数 factor_icir 返回标量
- 能力计数：strategies=40 / optimizers=28（两类目标封顶）
- 自驱动缺口生成器仍在工作（next_gaps 返回非空前缀）

纯 pandas/numpy，离线合成数据，零额外依赖。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from lianghua.strategy.registry import get_strategy, STRATEGY_NAMES
from lianghua.portfolio.registry import get_optimizer, OPTIMIZER_NAMES
from lianghua.factor.combine import factor_icir
from lianghua.core.capabilities import summary_counts
from lianghua.core import selfdrive


def _make_ohlcv(n=300, seed=11):
    np.random.seed(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), index=idx)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close.shift(1).fillna(close)
    vol = pd.Series(np.random.rand(n) * 1e6, index=idx)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _make_returns(n=200, seed=13):
    np.random.seed(seed)
    cols = ["A", "B", "C", "D", "E"]
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(np.random.randn(n, 5) * 0.01, index=idx, columns=cols)


def test_strategies_141_145():
    df = _make_ohlcv()
    n = len(df)
    for name in ["pivot_points", "heikin_ashi", "renko_trend", "demark", "klinger"]:
        s = get_strategy(name).generate_signals(df)
        uniq = set(pd.Series(s).unique().tolist())
        assert uniq.issubset({-1, 0, 1}), f"{name} 信号越界: {uniq}"
        assert len(s) == n, f"{name} 长度不对: {len(s)}"
    assert len(STRATEGY_NAMES) == 40, f"策略总数应为 40, 实得 {len(STRATEGY_NAMES)}"


def test_optimizers_146_149():
    R = _make_returns()
    for name in ["bayesian_shrinkage", "max_return", "min_track_error", "robust_cov"]:
        w = get_optimizer(name)(R)
        assert abs(w.sum() - 1.0) < 1e-6, f"{name} 权重和≠1: {w.sum()}"
        assert (w >= 0).all(), f"{name} 出现负权重: {w.min()}"
    assert len(OPTIMIZER_NAMES) == 28, f"优化器总数应为 28, 实得 {len(OPTIMIZER_NAMES)}"


def test_factor_icir_150():
    cols = ["A", "B", "C", "D", "E"]
    fp = pd.DataFrame(
        np.random.randn(60, 5),
        index=pd.date_range("2023-01-01", periods=60, freq="D"),
        columns=cols,
    )
    icir = factor_icir(fp)
    assert isinstance(icir, float), f"factor_icir 应返回 float, 实得 {type(icir)}"


def test_counts():
    c = summary_counts()
    assert c["strategies"] == 40, c
    assert c["optimizers"] == 28, c
    assert c["factor_functions"] >= 17, c


def test_selfdrive_next_gaps():
    gaps = selfdrive.next_gaps(3)
    assert len(gaps) > 0, "next_gaps 应为非空"
    assert all("name" in g and "category" in g for g in gaps), gaps
    # 策略/优化器已达目标，缺口应转向 perf/risk/indicators/factor
    cats = {g["category"] for g in gaps}
    assert cats.issubset({
        "strategies", "optimizers", "indicators",
        "perf_functions", "risk_functions", "factor_functions",
    }), cats


if __name__ == "__main__":
    test_strategies_141_145()
    test_optimizers_146_149()
    test_factor_icir_150()
    test_counts()
    test_selfdrive_next_gaps()
    print("ALL_ITER_141_150_OK")
