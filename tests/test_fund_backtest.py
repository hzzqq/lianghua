"""基金回测：NAV 守卫回归测试（迭代76）。

覆盖隐性正确性 bug：
- close 为 NaN/inf/<=0 时，原代码 shares=cost/nav 会把整个权益曲线污染为
  NaN/inf，BacktestResult.stats 会崩溃或静默产出错误指标（与 cycle37/74 同因）。
- 新增：无效净值跳过交易，沿用上一刻权益，持仓按最近有效净值标记。
"""
import math

import numpy as np
import pandas as pd
import pytest

from lianghua.fund.backtest import FundBacktest, DCABacktest


def _df(closes):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"date": idx.astype(str), "close": closes})


def test_fund_backtest_normal_finite():
    df = _df([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    sig = pd.Series(1, index=df.index)  # 全程持有
    res = FundBacktest(init_cash=1_000_000.0).run(df, sig, "FUND.X")
    assert np.all(np.isfinite(res.equity.values))
    assert len(res.trades) == 1  # 仅一次建仓
    # 末值 = 现金(0) + 份额 * 末净值，应为有限正数
    assert res.equity.iloc[-1] > 0


def test_fund_backtest_nan_nav_skips_and_stays_finite():
    # 第 4 根 K 线净值为 NaN：不应触发交易、权益曲线全程有限
    closes = [10.0, 11.0, 12.0, np.nan, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    df = _df(closes)
    sig = pd.Series(1, index=df.index)
    res = FundBacktest(init_cash=1_000_000.0).run(df, sig, "FUND.X")
    assert np.all(np.isfinite(res.equity.values))
    # NaN 那根权益沿用上一刻（建仓在 bar0，bar3 跳过，权益==前一有效值）
    assert res.equity.iloc[3] == res.equity.iloc[2]
    # 不含以 NaN 净值为成交价的交易
    bad = [t for t in res.trades if not math.isfinite(t["nav"])]
    assert bad == []


def test_fund_backtest_zero_nav_guarded():
    closes = [10.0, 11.0, 0.0, 13.0, 14.0, 15.0]  # 中间出现 0 净值
    df = _df(closes)
    sig = pd.Series(1, index=df.index)
    res = FundBacktest(init_cash=1_000_000.0).run(df, sig, "FUND.X")
    assert np.all(np.isfinite(res.equity.values))


def test_dca_nan_nav_skips_bar():
    closes = [10.0, 11.0, np.nan, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    df = _df(closes)
    res = DCABacktest(init_cash=1_000_000.0, amount=10_000.0, period=2).run(df, "FUND.X")
    assert np.all(np.isfinite(res.equity.values))
    # NaN 那根权益沿用上一刻（未在该 bar 投入）
    assert res.equity.iloc[2] == res.equity.iloc[1]
    bad = [t for t in res.trades if not math.isfinite(t["nav"])]
    assert bad == []


def test_dca_normal_accumulates():
    closes = list(np.linspace(10, 20, 10))
    df = _df(closes)
    res = DCABacktest(init_cash=1_000_000.0, amount=10_000.0, period=3).run(df, "FUND.X")
    assert np.all(np.isfinite(res.equity.values))
    assert len(res.trades) >= 1
