"""risk/ 打磨：止损(ATR/硬)、止盈(分段减仓修复)、区制(平滑) 覆盖与守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.risk.stop_loss import (
    trailing_stop, hard_stop, atr_trailing_stop,
    average_true_range, true_range, stop_triggered,
)
from lianghua.risk.take_profit import take_profit_levels, scaled_exit
from lianghua.risk.regime import detect_regime, smooth_regime, regime_stats


def _ohlc(n=120, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(high, index=idx, name="high"), \
        pd.Series(low, index=idx, name="low"), \
        pd.Series(close, index=idx, name="close")


def test_hard_stop_bounds():
    assert hard_stop(100, 0.1) == pytest.approx(90.0)
    with pytest.raises(ValueError):
        hard_stop(100, 0.0)


def test_atr_helpers_finite():
    high, low, close = _ohlc()
    tr = true_range(high, low, close)
    atr = average_true_range(high, low, close, period=14)
    assert np.all(np.isfinite(tr.dropna()))
    assert np.all(np.isfinite(atr.dropna()))
    assert (atr.dropna() > 0).all()


def test_atr_trailing_stop_follows_highs():
    high, low, close = _ohlc()
    atr = average_true_range(high, low, close)
    stop = atr_trailing_stop(close, atr, mult=3.0)
    # 吊灯止损绝不应高于运行最高点（不会把止损设在最高价之上）
    assert (stop <= close.cummax()).all()
    assert np.all(np.isfinite(stop.dropna()))


def test_trailing_stop_floor():
    price = pd.Series([100, 110, 105, 95])
    stop = trailing_stop(price, entry=100, init_stop=0.05, trail_pct=0.1)
    # 硬止损下限 95
    assert (stop >= 95.0).all()
    # 价格最高点110时止损=99
    assert stop.iloc[1] == pytest.approx(99.0)


def test_take_profit_levels_length_guard():
    with pytest.raises(ValueError):
        take_profit_levels(100, [0.1, 0.2], [0.5])


def test_scaled_exit_honors_ratios():
    # 价格 100→108→112→122→132：+10%(=110) 减 0.3，+20%(=120) 再减 0.7 → 累计 1.0
    price = pd.Series([100, 108, 112, 122, 132])
    out = scaled_exit(price, entry=100, levels=[0.1, 0.2], ratios=[0.3, 0.7])
    assert out.iloc[0] == 0.0          # 100 < 110 未触及
    assert out.iloc[1] == 0.0          # 108 < 110 未触及
    assert out.iloc[2] == pytest.approx(0.3)   # 112 >= 110 触及 +10%
    assert out.iloc[3] == pytest.approx(1.0)   # 122 同时触及 +20%，累计满仓退出
    assert out.iloc[-1] == pytest.approx(1.0)
    # 单调不减
    assert (out.diff().dropna() >= 0).all()


def test_scaled_exit_default_ratios():
    price = pd.Series([100, 105, 115, 125])
    out = scaled_exit(price, entry=100, levels=[0.05, 0.15, 0.25])
    assert out.iloc[-1] == pytest.approx(1.0)  # 默认末档补足到 1


def test_regime_detect_and_smooth():
    rng = np.random.default_rng(7)
    # 前段低波动，后段高波动
    r = pd.Series(np.concatenate([
        rng.normal(0, 0.001, 60),
        rng.normal(0, 0.02, 60),
    ]))
    reg = detect_regime(r, slow=30, mult=1.5)
    assert set(np.unique(reg.dropna())).issubset({0, 1})
    sm = smooth_regime(reg, min_hold=5)
    # 平滑后仍为 0/1 序列，且长度一致
    assert len(sm) == len(reg)
    assert set(np.unique(sm)).issubset({0, 1})
    # 平滑不应引入比原始更多切换（闸门抑制抖动）
    raw_switches = int((reg.diff().abs() > 0).sum())
    sm_switches = int((sm.diff().abs() > 0).sum())
    assert sm_switches <= raw_switches


def test_regime_stats_handles_nan():
    r = pd.Series([0.01, np.nan, 0.02, -0.01, 0.0])
    reg = pd.Series([0, 0, 1, 1, 0])
    stats = regime_stats(r, reg)
    assert "mean" in stats.columns
    assert stats["count"].sum() == 4  # NaN 行被剔除
