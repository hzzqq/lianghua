"""基金筛选器：新增 method/top/explain + 隐性非有限污染修复回归测试。

覆盖：
- 非有限因子值(inf/nan)不再污染整体排名（关键隐性 bug）
- 动量基准 <=0 不再产生 inf
- lookback / 空数据 / 权重 / method 守卫
- method 切换、top()、explain() 新能力
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.fund.screener import FundScreener


def _nav(seed, drift, vol, n=120):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n)
    return pd.Series(1000.0 * np.cumprod(1.0 + ret))


def _make_df():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    A = _nav(1, 0.0015, 0.004)          # 稳健上行
    B = _nav(2, 0.0010, 0.004).copy()
    B.iloc[60] = 0.0                     # 中途净值为 0 -> 收益 inf / 回撤异常
    C = pd.Series(1000.0, index=idx)    # 完全平坦
    return pd.DataFrame({"A": A, "B": B, "C": C}, index=idx)


def test_screen_returns_ranking():
    sc = FundScreener.from_frame(_make_df(), lookback=60)
    out = sc.screen()
    assert "综合分" in out.columns
    assert "排名" in out.columns
    assert list(out["排名"]) == list(range(1, len(out) + 1))
    # 排名按综合分降序
    assert out["综合分"].is_monotonic_decreasing


def test_nonfinite_fund_does_not_poison_ranking():
    sc = FundScreener.from_frame(_make_df(), lookback=60)
    out = sc.screen()
    # 关键回归：含 0 净值(inf/nan 因子)的基金不得让整体综合分变 NaN
    assert out["综合分"].notna().all()
    assert np.isfinite(out["综合分"]).all()


def test_momentum_nonpositive_base_sanitized():
    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    # 净值从 0 起步（基准<=0）不应产生 inf 动量
    s = pd.Series([0.0] + list(np.linspace(1, 5, 79)), index=idx)
    df = pd.DataFrame({"Z": s})
    sc = FundScreener.from_frame(df, lookback=30)
    out = sc.screen()
    assert np.isfinite(out.loc["Z", "综合分"])


def test_minmax_method_runs():
    sc = FundScreener.from_frame(_make_df(), lookback=60)
    out = sc.screen(method="minmax")
    assert np.isfinite(out["综合分"]).all()


def test_invalid_method_raises():
    sc = FundScreener.from_frame(_make_df(), lookback=60)
    with pytest.raises(ValueError):
        sc.screen(method="bogus")


def test_lookback_guard():
    with pytest.raises(ValueError):
        FundScreener(lookback=0)
    with pytest.raises(ValueError):
        FundScreener(lookback=-1)


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        FundScreener.from_frame(pd.DataFrame())
    with pytest.raises(ValueError):
        FundScreener.from_series({})


def test_all_zero_weights_raise():
    sc = FundScreener.from_frame(_make_df(), lookback=60)
    with pytest.raises(ValueError):
        sc.screen(weights={"momentum": 0, "volatility": 0, "max_drawdown": 0, "sharpe": 0})


def test_top_returns_n_rows():
    sc = FundScreener.from_frame(_make_df(), lookback=60)
    top = sc.top(2)
    assert len(top) == 2
    assert list(top["排名"]) == [1, 2]


def test_top_invalid_n_raises():
    sc = FundScreener.from_frame(_make_df(), lookback=60)
    with pytest.raises(ValueError):
        sc.top(0)


def test_explain_returns_factors():
    sc = FundScreener.from_frame(_make_df(), lookback=60)
    exp = sc.explain("A")
    for f in FundScreener.FACTORS:
        assert f in exp
    assert "综合分" in exp
    assert "排名" in exp


def test_explain_unknown_symbol_raises():
    sc = FundScreener.from_frame(_make_df(), lookback=60)
    with pytest.raises(KeyError):
        sc.explain("NOPE")
