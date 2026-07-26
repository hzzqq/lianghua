"""技术指标（indicators）：输入守卫补全 + crossover/crossunder 信号原语 + zigzag 校验。

续做轮 56（indicators 集群，覆盖 tech2/tech3/tech4 此前未打磨模块）：
- 隐性修复：tech2/tech3/tech4 中 DataFrame 指标此前直接下标取列，缺列时抛难定位的
  KeyError；现统一用 tech._require_cols / _check_period 给出清晰错误，并校验周期/参数。
- 隐性修复：zigzag(pct<=0) 会退化为逐根翻转，现显式拒绝非正 pct。
- 新需求：tech.crossover / crossunder 黄金/死亡交叉布尔信号原语，供策略复用。
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.indicators import tech, tech2, tech3, tech4


@pytest.fixture
def ohlcv():
    n = 40
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    vol = rng.integers(1e5, 1e6, n).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": vol})


def test_missing_column_raises_clear_error(ohlcv):
    bad = ohlcv.drop(columns=["high"])
    with pytest.raises(ValueError):
        tech2.supertrend(bad)
    with pytest.raises(ValueError):
        tech3.keltner_channels(bad)
    with pytest.raises(ValueError):
        tech4.adx(bad)
    with pytest.raises(ValueError):
        tech4.heikin_ashi(bad.drop(columns=["open"]))


def test_invalid_period_raises(ohlcv):
    with pytest.raises(ValueError):
        tech2.aroon(ohlcv, period=0)
    with pytest.raises(ValueError):
        tech3.hull_moving_average(ohlcv["close"], period=-1)
    with pytest.raises(ValueError):
        tech4.know_sure_thing(ohlcv, r1=0)


def test_indicators_run_and_length(ohlcv):
    for fn in (tech2.supertrend, tech3.keltner_channels, tech3.donchian_channel,
               tech3.chandelier_exit, tech4.adx, tech4.pivot, tech4.heikin_ashi,
               tech4.price_channels, tech3.mass_index, tech4.chaikin_volatility):
        out = fn(ohlcv)
        assert len(out) == len(ohlcv)


def test_zigzag_rejects_nonpositive_pct(ohlcv):
    with pytest.raises(ValueError):
        tech4.zigzag(ohlcv, pct=0)
    with pytest.raises(ValueError):
        tech4.zigzag(ohlcv, pct=-0.1)
    # 正常 pct 返回等长布尔方向
    zz = tech4.zigzag(ohlcv, pct=0.02)
    assert len(zz) == len(ohlcv) and set(zz.unique()).issubset({-1, 0, 1})


def test_crossover_crossunder(ohlcv):
    fast = ohlcv["close"].rolling(5).mean()
    slow = ohlcv["close"].rolling(20).mean()
    up = tech.crossover(fast, slow)
    dn = tech.crossunder(fast, slow)
    assert len(up) == len(ohlcv)
    # 交叉点不能在同一根同时为金叉与死叉
    assert not (up & dn).any()
    # 首值必为 False（无前一根可比）
    assert not up.iloc[0] and not dn.iloc[0]
