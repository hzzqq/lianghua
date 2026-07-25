"""向量化回测测试（Round 16）：新增成交流易反推 + 输入守卫（缺失列/非正价/空数据/负成本）。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.backtest.vectorized import (
    vectorized_backtest, positions_from_signals, extract_trades,
)


def _make_df(prices, idx=None):
    if idx is None:
        idx = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"close": prices}, index=idx)


def test_basic_long_short_backtest():
    df = _make_df([100, 102, 101, 105, 107])
    sig = pd.Series([1, 1, 0, -1, -1], index=df.index)
    res = vectorized_backtest(df, sig)
    assert res["equity"].iloc[0] == 1_000_000.0
    assert "sharpe" in res["metrics"]
    assert res["metrics"]["num_trades"] >= 1  # 至少一进一出


def test_positions_next_day_shift():
    sig = pd.Series([1, 1, -1, 0, 0])
    pos = positions_from_signals(sig, shift=True)
    assert pos.iloc[0] == 0  # 首个信号次日才生效，无前视


def test_extract_trades_round_trip():
    pos = pd.Series([0, 1, 1, 0, -1, -1, 0])
    trades = extract_trades(pos)
    # 一笔多头往返 + 一笔空头往返
    assert len(trades) == 2
    assert trades[0]["side"] == "long"
    assert trades[1]["side"] == "short"
    assert trades[0]["exit"] == 3


def test_missing_close_column_raises():
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError):
        vectorized_backtest(df, pd.Series([1, 1, 1]))


def test_non_positive_price_no_inf():
    df = _make_df([100, 0, 0, 105, 107])  # 0 价会被前向填充
    sig = pd.Series([1, 1, 1, 1, 1], index=df.index)
    res = vectorized_backtest(df, sig)
    assert not np.any(np.isinf(res["daily_returns"]))
    assert not np.any(np.isnan(res["equity"].to_numpy()))


def test_empty_df_safe():
    df = _make_df([])
    res = vectorized_backtest(df, pd.Series([], dtype=float))
    assert res["metrics"]["num_trades"] == 0
    assert not np.any(np.isnan(res["equity"].to_numpy())) if len(res["equity"]) else True


def test_negative_cost_clamped():
    df = _make_df([100, 102, 101, 105])
    sig = pd.Series([1, 1, -1, -1], index=df.index)
    res = vectorized_backtest(df, sig, cost_rate=-0.01)
    # 不应崩溃，且成本被夹为 0
    assert np.isfinite(res["equity"].iloc[-1])


def test_short_profit():
    df = _make_df([100, 90, 80])  # 下跌行情
    sig = pd.Series([-1, -1, -1], index=df.index)
    res = vectorized_backtest(df, sig)
    # 做空下跌应盈利
    assert res["equity"].iloc[-1] > 1_000_000.0
