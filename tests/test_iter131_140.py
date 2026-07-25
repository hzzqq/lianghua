"""迭代 131–140（第八轮）集成测试。

覆盖：
- 5 个新策略（经统一注册表，输出 {-1,0,1}）
- 2 个新优化器（经注册表，权重和≈1）
- 绩效比率 omega/calmar/tail_ratio
- 风险尾部 tail_dependence / risk_contribution
- 因子 factor_neutrality / factor_turnover
- 能力计数与自驱动缺口生成器
全部离线条形/随机数据，零网络。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lianghua.strategy.registry import get_strategy, STRATEGY_NAMES  # noqa: E402
from lianghua.portfolio.registry import get_optimizer, OPTIMIZER_NAMES  # noqa: E402
from lianghua.core.capabilities import summary_counts  # noqa: E402
from lianghua.core import selfdrive  # noqa: E402
import lianghua.perf.ratios as ratios  # noqa: E402
import lianghua.risk.tail as tail  # noqa: E402
import lianghua.factor.combine as combine  # noqa: E402


def _ohlc(n=300, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(rng.standard_normal(n)), index=idx)
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    open_ = close.shift(1).fillna(close)
    vol = pd.Series(rng.random(n) * 1e6, index=idx)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _rets(n=300, k=5, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        rng.standard_normal((n, k)) * 0.01,
        index=idx,
        columns=[f"A{i}" for i in range(k)],
    )


# ---------- 策略 ----------
NEW_STRATS = ["parabolic_sar", "adx_trend", "cci_signal", "roc", "ultimate_oscillator"]


def test_new_strategies_via_registry():
    df = _ohlc()
    for name in NEW_STRATS:
        assert name in STRATEGY_NAMES, f"{name} 未注册"
        sig = get_strategy(name).generate_signals(df)
        assert len(sig) == len(df), name
        assert set(np.unique(sig)).issubset({-1, 0, 1}), (name, set(np.unique(sig)))
    # 计数
    assert len(STRATEGY_NAMES) == 40


# ---------- 优化器 ----------
NEW_OPTS = ["min_tail_risk", "vol_target_opt"]


def test_new_optimizers_via_registry():
    rets = _rets()
    for name in NEW_OPTS:
        assert name in OPTIMIZER_NAMES, f"{name} 未注册"
        w = get_optimizer(name)(rets)
        assert isinstance(w, pd.Series)
        assert w.index.tolist() == list(rets.columns)
        assert abs(w.sum() - 1.0) < 1e-6, (name, w.sum())
        assert (w >= -1e-9).all()
    assert len(OPTIMIZER_NAMES) == 28


# ---------- 绩效比率 ----------
def test_perf_ratios():
    eq = pd.Series(100 * np.cumprod(1 + _rets(n=200).mean(axis=1)))
    r = _rets(n=200).mean(axis=1)
    assert np.isfinite(ratios.omega_ratio(r))
    assert np.isfinite(ratios.calmar_ratio(eq))
    assert np.isfinite(ratios.tail_ratio(r))


# ---------- 风险尾部 ----------
def test_risk_tail():
    rets = _rets()
    td = tail.tail_dependence(rets["A0"], rets["A1"])
    assert 0.0 <= td <= 1.0, td
    rc = tail.risk_contribution(rets, np.ones(rets.shape[1]) / rets.shape[1])
    assert isinstance(rc, pd.Series)
    assert rc.index.tolist() == list(rets.columns)
    assert np.all(np.isfinite(rc.values))


# ---------- 因子 ----------
def test_factor_funcs():
    rng = np.random.default_rng(11)
    panel = pd.DataFrame(
        rng.standard_normal((120, 8)),
        index=pd.date_range("2023-01-01", periods=120, freq="D"),
        columns=[f"S{i}" for i in range(8)],
    )
    # 中性化：剔除一个保护变量
    protect = pd.DataFrame(
        rng.standard_normal((120, 8)),
        index=panel.index, columns=panel.columns,
    )
    neut = combine.factor_neutrality(panel.iloc[0], protect.iloc[0])
    assert isinstance(neut, pd.Series) and len(neut) == 8
    # 面板中性化
    neut_panel = combine.factor_neutrality(panel, protect)
    assert neut_panel.shape == panel.shape
    # 换手率
    to = combine.factor_turnover(panel)
    assert 0.0 <= to <= 1.0, to


# ---------- 能力计数 ----------
def test_capability_counts():
    c = summary_counts()
    assert c["strategies"] == 40
    assert c["optimizers"] == 28
    assert c["indicators"] == 37
    assert c["perf_functions"] >= 42
    assert c["risk_functions"] >= 34
    assert c["factor_functions"] >= 16


# ---------- 自驱动缺口生成器 ----------
def test_selfdrive_gaps():
    gaps = selfdrive.next_gaps(10)
    assert isinstance(gaps, list) and len(gaps) > 0
    d = selfdrive.deficit()
    # 本轮已实现项不应还出现在 pending 中
    for cat in ("strategies", "optimizers", "perf_functions", "risk_functions", "factor_functions"):
        pend = d[cat]["pending"]
        for done in ("parabolic_sar", "min_tail_risk", "omega_ratio",
                     "tail_dependence", "factor_neutrality"):
            assert done not in pend, (cat, done)


if __name__ == "__main__":
    test_new_strategies_via_registry()
    test_new_optimizers_via_registry()
    test_perf_ratios()
    test_risk_tail()
    test_factor_funcs()
    test_capability_counts()
    test_selfdrive_gaps()
    print("ALL_ITER131_140_OK")
