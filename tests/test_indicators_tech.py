"""技术指标模块测试：正确性 + 输入守卫 + 新增能力（Stochastic / Williams %R）。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.indicators import tech


def _ohlc(n=120, seed=1):
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n))
    high = close + rng.rand(n) * 2
    low = close - rng.rand(n) * 2
    volume = rng.randint(1e4, 1e5, n).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": volume})


def test_rsi_basic():
    s = pd.Series(np.random.RandomState(0).randn(100).cumsum() + 100)
    r = tech.rsi(s)
    assert len(r) == len(s)
    assert r.iloc[10:].between(0, 100).all()


def test_rsi_flat_is_neutral():
    # 平坦行情 RSI 应中性化为 50，而非错误归零。
    flat = pd.Series([100.0] * 60)
    r = tech.rsi(flat, period=14)
    assert (r.dropna() == 50.0).all()


def test_rsi_strictly_rising_near_100():
    up = pd.Series(np.arange(1, 61, dtype=float))
    r = tech.rsi(up, period=14)
    assert r.dropna().iloc[-1] > 99


def test_rsi_invalid_period():
    with pytest.raises(ValueError):
        tech.rsi(pd.Series(np.arange(10.0)), period=0)
    with pytest.raises(ValueError):
        tech.rsi(pd.Series(np.arange(10.0)), period=-3)


def test_rsi_non_series_raises():
    # 二维数组无法构成 1-D 序列，应触发守卫错误。
    with pytest.raises((TypeError, ValueError)):
        tech.rsi(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_rsi_empty_raises():
    with pytest.raises(ValueError):
        tech.rsi(pd.Series([], dtype=float))


def test_macd_shape_and_range():
    df = _ohlc()
    m = tech.macd(df["close"])
    assert list(m.columns) == ["dif", "dea", "hist"]
    assert len(m) == len(df)


def test_kdj_missing_cols_raises():
    bad = pd.DataFrame({"close": np.arange(10.0)})
    with pytest.raises(ValueError):
        tech.kdj(bad)


def test_atr_missing_cols_raises():
    bad = pd.DataFrame({"close": np.arange(10.0)})
    with pytest.raises(ValueError):
        tech.atr(bad)


def test_cci_basic():
    df = _ohlc()
    c = tech.cci(df, period=20)
    assert len(c) == len(df)


def test_obv_requires_volume():
    bad = pd.DataFrame({"close": np.arange(10.0)})
    with pytest.raises(ValueError):
        tech.obv(bad)


def test_vwap_zero_volume_guard():
    df = _ohlc()
    df.loc[df.index[0], "volume"] = 0.0  # 首行成交量为 0
    v = tech.vwap(df)
    # 除首行 NaN 外，其余应为有限数，不出现 inf。
    assert np.isfinite(v.iloc[1:]).all()


def test_stochastic_bounds():
    df = _ohlc()
    st = tech.stochastic(df, n=14)
    assert list(st.columns) == ["k", "d"]
    # 慢速 %K/%D 落在 [0,100]。
    assert ((st.dropna() >= 0) & (st.dropna() <= 100)).all().all()


def test_williams_r_bounds():
    df = _ohlc()
    wr = tech.williams_r(df, period=14)
    assert wr.dropna().between(-100, 0).all()


def test_new_indicators_registered():
    assert "stochastic" in tech.__dict__
    assert "williams_r" in tech.__dict__
