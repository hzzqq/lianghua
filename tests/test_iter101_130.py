"""迭代 101-130 集成测试（第七轮扩展能力）。

覆盖：
  - 10 个新技术指标 (lianghua/indicators/tech2.py)
  - 10 个新策略（经注册表 get_strategy 调用，输出 {-1,0,1}）
  - 5 个新高级优化器（经注册表 get_optimizer 调用，权重和≈1）
  - 新增绩效/风险/因子函数 (perf/ratios, risk/tail, factor/combine)
  - 能力索引计数（strategies=30, optimizers=22, indicators=18）

全部离线、纯 pandas/numpy 合成数据验证，零网络依赖。
运行：python -m pytest tests/test_iter101_130.py -q
或直接：python tests/test_iter101_130.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N = 260


def _ohlcv(n=N, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _returns_df(cols=5, n=N, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    data = {f"A{i}": rng.normal(0.0005, 0.015 + 0.003 * i, n) for i in range(cols)}
    return pd.DataFrame(data, index=idx)


def _equity(n=N, seed=3):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, 0.018, n)
    return pd.Series(100.0 * np.exp(np.cumsum(ret)),
                     index=pd.date_range("2023-01-01", periods=n, freq="B"))


# ---------------------------------------------------------------- 指标 ×10
def test_indicators_tech2():
    from lianghua.indicators import tech2 as t2
    df = _ohlcv()
    close = df["close"]

    checks = {
        "supertrend": t2.supertrend(df),
        "aroon": t2.aroon(df),
        "vortex": t2.vortex(df),
        "trix": t2.trix(close),
        "williams_r": t2.williams_r(df),
        "cmf": t2.cmf(df),
        "mfi": t2.mfi(df),
        "stoch_rsi": t2.stoch_rsi(close),
        "dpo": t2.dpo(close),
        "ppo": t2.ppo(close),
    }
    for name, out in checks.items():
        assert len(out) == N, f"{name} 长度应为 {N}, 实得 {len(out)}"
        assert isinstance(out, (pd.Series, pd.DataFrame)), f"{name} 类型异常"
        # 至少有有效数值
        vals = out.to_numpy().ravel()
        assert np.isfinite(vals[~np.isnan(vals)]).all(), f"{name} 含非有限值"
    # williams_r 范围 [-100,0]
    wr = t2.williams_r(df).dropna()
    assert (wr <= 0.001).all() and (wr >= -100.001).all()
    # mfi 范围 [0,100]
    mfi = t2.mfi(df).dropna()
    assert (mfi >= -0.001).all() and (mfi <= 100.001).all()
    print("indicators tech2: 10/10 OK")


# ---------------------------------------------------------------- 策略 ×10
NEW_STRATEGIES = [
    "supertrend", "aroon", "vortex", "trix", "williams_r",
    "elder_ray", "ichimoku", "macd_hist", "chaikin", "stoch_rsi",
]


def test_new_strategies_via_registry():
    from lianghua.strategy.registry import get_strategy, STRATEGY_NAMES
    assert len(STRATEGY_NAMES) == 40, f"策略总数应为 40, 实得 {len(STRATEGY_NAMES)}"
    df = _ohlcv()
    for name in NEW_STRATEGIES:
        assert name in STRATEGY_NAMES, f"{name} 未注册"
        strat = get_strategy(name)
        sig = strat.generate_signals(df.copy())
        assert isinstance(sig, pd.Series), f"{name} 信号非 Series"
        assert len(sig) == N, f"{name} 信号长度 {len(sig)} != {N}"
        uniq = set(pd.unique(sig.dropna()))
        assert uniq.issubset({-1, 0, 1, -1.0, 0.0, 1.0}), f"{name} 信号越界: {uniq}"
    print("new strategies via registry: 10/10 OK")


# ---------------------------------------------------------------- 优化器 ×5
NEW_OPTIMIZERS = ["max_sharpe", "min_cvar", "shrinkage_min_var", "max_entropy", "momentum_score"]


def test_new_optimizers_via_registry():
    from lianghua.portfolio.registry import get_optimizer, OPTIMIZER_NAMES
    assert len(OPTIMIZER_NAMES) == 28, f"优化器总数应为 28, 实得 {len(OPTIMIZER_NAMES)}"
    rets = _returns_df()
    for name in NEW_OPTIMIZERS:
        assert name in OPTIMIZER_NAMES, f"{name} 未注册"
        opt = get_optimizer(name)
        w = opt(rets)
        assert isinstance(w, pd.Series), f"{name} 权重非 Series"
        assert len(w) == rets.shape[1], f"{name} 权重维度错"
        assert (w >= -1e-6).all(), f"{name} 存在负权重"
        assert abs(float(w.sum()) - 1.0) < 1e-3, f"{name} 权重和 {float(w.sum())} != 1"
    print("new optimizers via registry: 5/5 OK")


# ---------------------------------------------------------------- 绩效 perf/ratios
def test_perf_ratios():
    from lianghua.perf import ratios as R
    eq = _equity()
    bench = _equity(seed=99)
    assert np.isfinite(R.ulcer_index(eq))
    assert np.isfinite(R.martin_ratio(eq))
    assert np.isfinite(R.gain_to_pain(eq))
    assert np.isfinite(R.cagr(eq))
    assert np.isfinite(R.k_ratio(eq))
    cap = R.up_down_capture(eq, bench)
    assert set(cap.keys()) == {"up_capture", "down_capture"}
    assert R.ulcer_index(eq) >= 0
    print("perf.ratios: 6/6 OK")


# ---------------------------------------------------------------- 风险 risk/tail
def test_risk_tail():
    from lianghua.risk import tail as T
    rets = _returns_df()
    eq = _equity()
    w = pd.Series(np.repeat(1.0 / rets.shape[1], rets.shape[1]), index=rets.columns)
    # 零依赖正态分位近似
    assert abs(T._norm_ppf(0.05) - (-1.6449)) < 1e-2
    dd = T.downside_deviation(rets["A0"])
    assert dd >= 0 and np.isfinite(dd)
    cvar = T.component_var(rets, w)
    assert isinstance(cvar, pd.Series) and len(cvar) == rets.shape[1]
    assert np.isfinite(cvar.to_numpy()).all()
    cdar = T.cdar(eq)
    assert cdar >= 0 and np.isfinite(cdar)
    print("risk.tail: 4/4 OK")


# ---------------------------------------------------------------- 因子 factor/combine
def test_factor_combine():
    from lianghua.factor import combine as C
    idx = pd.date_range("2023-01-01", periods=50, freq="B")
    rng = np.random.default_rng(5)
    f1 = pd.Series(rng.normal(size=50), index=idx)
    f2 = pd.Series(rng.normal(size=50), index=idx)
    comb = C.factor_combine({"f1": f1, "f2": f2}, weights={"f1": 0.6, "f2": 0.4})
    assert isinstance(comb, pd.Series) and len(comb) == 50
    assert np.isfinite(comb.to_numpy()).all()
    # rank 法
    comb2 = C.factor_combine({"f1": f1, "f2": f2}, method="rank")
    assert len(comb2) == 50
    panel = pd.DataFrame(rng.normal(size=(50, 6)), index=idx)
    ac = C.factor_autocorr(panel, lag=1)
    assert np.isfinite(ac)
    print("factor.combine: OK")


# ---------------------------------------------------------------- 能力索引
def test_capabilities_counts():
    from lianghua.core.capabilities import summary_counts
    c = summary_counts()
    assert c["strategies"] == 40, c
    assert c["optimizers"] == 28, c
    assert c["indicators"] == 37, c
    print("capabilities counts: OK", {k: c[k] for k in ("strategies", "optimizers", "indicators")})


def _run_all():
    fns = [
        test_indicators_tech2,
        test_new_strategies_via_registry,
        test_new_optimizers_via_registry,
        test_perf_ratios,
        test_risk_tail,
        test_factor_combine,
        test_capabilities_counts,
    ]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
    print(f"\n=== iter101-130 集成测试全部通过: {passed}/{len(fns)} ===")


if __name__ == "__main__":
    _run_all()
