"""backtest/engine 与 vectorized 打磨测试：NaN/非正价格与缺 date 列的鲁棒性 + 新增能力。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lianghua.backtest.engine import BacktestEngine, BacktestResult
from lianghua.backtest.vectorized import vectorized_backtest


def _idx(n):
    return pd.date_range("2023-01-01", periods=n, freq="D")


def test_run_skips_nan_price():
    prices = [100, 101, np.nan, 103, 104]
    df = pd.DataFrame({"date": _idx(5), "close": prices})
    sig = pd.Series([1, 0, 0, 0, -1], index=_idx(5))
    res = BacktestEngine().run(df, sig)
    assert len(res.equity) == 5
    assert res.skipped >= 1
    assert np.all(np.isfinite(res.equity.to_numpy()))
    assert res.trades  # 仍完成买卖


def test_run_skips_zero_price():
    prices = [100, 0, 0, 105, 107]
    df = pd.DataFrame({"date": _idx(5), "close": prices})
    sig = pd.Series([1, 0, 0, 0, -1], index=_idx(5))
    res = BacktestEngine().run(df, sig)
    assert res.skipped >= 1
    assert np.all(np.isfinite(res.equity.to_numpy()))


def test_run_date_index_without_date_column():
    # df 无 'date' 列（仅日期索引）时不应 KeyError
    prices = [100, 101, 102, 103, 104]
    df = pd.DataFrame({"close": prices}, index=_idx(5))
    sig = pd.Series([1, 0, 0, 0, -1], index=_idx(5))
    res = BacktestEngine().run(df, sig)
    assert len(res.equity) == 5
    assert np.all(np.isfinite(res.equity.to_numpy()))


def test_run_raises_on_missing_close():
    with pytest.raises(ValueError):
        BacktestEngine().run(pd.DataFrame({"open": [1, 2, 3]}), pd.Series([1, 0, -1]))


def test_stats_nan_first_safe():
    res = BacktestResult(pd.Series([np.nan, 101.0, 102.0]), [], pd.Series([0, 0, 0]), None)
    assert res.stats()["total_return"] == 0.0


def test_result_returns():
    eq = pd.Series([100.0, 110.0, 99.0])
    res = BacktestResult(eq, [], pd.Series([0, 0, 0]), None)
    r = res.returns()
    assert len(r) == 3
    assert r.iloc[0] == 0.0  # fillna
    assert r.iloc[1] == pytest.approx(0.1)


def test_vectorized_nan_price_masked():
    df = pd.DataFrame({"close": [100, np.nan, 105, 107]}, index=_idx(4))
    sig = pd.Series([1, 1, -1, -1], index=_idx(4))
    res = vectorized_backtest(df, sig)
    assert np.all(np.isfinite(res["equity"].to_numpy()))
    assert not np.any(np.isinf(res["daily_returns"].to_numpy()))
