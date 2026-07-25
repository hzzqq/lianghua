"""因子引擎测试（Round 19）：新增 evaluate_all + 隐性修复（回测 NaN / qcut 崩溃 / 除零）。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.factor.engine import FactorEngine


def _price(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(0, 1, n))


def _df(n=300, seed=3, with_vol=True):
    close = _price(n, seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    d = {"close": close}
    if with_vol:
        rng = np.random.default_rng(seed + 1)
        d["volume"] = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(d, index=idx)


def test_compute_all_factors_finite():
    fe = FactorEngine(_df())
    for name in fe.FACTORS:
        f = fe.compute(name)
        # 除零修复后不应出现 inf
        assert not np.any(np.isinf(f.to_numpy())), f"因子 {name} 出现 inf"


def test_backtest_factor_not_all_nan():
    # 旧实现因 shift(1) 首行 NaN 导致整条净值 NaN
    fe = FactorEngine(_df())
    f = fe.compute("momentum_20")
    eq = fe.backtest_factor(f)
    assert eq.notna().any()
    assert np.isfinite(eq.dropna().iloc[-1]) if eq.notna().any() else True


def test_evaluate_low_variance_factor_no_crash():
    fe = FactorEngine(_df())
    const = pd.Series(1.0, index=fe.close.index)  # 常数因子
    res = fe.evaluate(const)
    assert "ic" in res  # 不抛 qcut 错误


def test_evaluate_forward_guard():
    fe = FactorEngine(_df())
    with pytest.raises(ValueError):
        fe.evaluate(fe.compute("momentum_20"), forward=0)


def test_backtest_top_quantile_guard():
    fe = FactorEngine(_df())
    with pytest.raises(ValueError):
        fe.backtest_factor(fe.compute("momentum_20"), top_quantile=0)


def test_evaluate_all_returns_sorted_df():
    fe = FactorEngine(_df())
    rep = fe.evaluate_all(forward=5)
    assert isinstance(rep, pd.DataFrame)
    assert len(rep) == len(fe.FACTORS)
    assert list(rep.columns)  # 含 factor/ic/abs_ic/long_short_return/n
    # abs_ic 降序（忽略 NaN 位置）
    vals = rep["abs_ic"].dropna().to_numpy()
    if len(vals) >= 2:
        assert vals[0] >= vals[-1] - 1e-9


def test_boll_pctb_zero_vol_safe():
    # 构造零波动段：boll 分母为 0 不应产生 inf
    close = np.concatenate([np.full(30, 100.0), _price(50)])
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D")
    fe = FactorEngine(pd.DataFrame({"close": close}, index=idx))
    f = fe.compute("boll_pctb")
    assert not np.any(np.isinf(f.to_numpy()))


def test_high_low_20_constant_range_safe():
    close = np.full(60, 50.0)
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D")
    fe = FactorEngine(pd.DataFrame({"close": close}, index=idx))
    f = fe.compute("high_low_20")
    assert not np.any(np.isinf(f.to_numpy()))
