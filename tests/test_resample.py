"""resample_ohlcv / resample_vwap 守卫 + 新能力回归。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.data.resample import resample_ohlcv, resample_vwap


def _mk(periods=30, vol_name="volume"):
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    rng = np.random.default_rng(0)
    px = 100 + rng.normal(0, 1, periods).cumsum()
    df = pd.DataFrame({
        "open": px + rng.normal(0, 0.1, periods),
        "high": px + 1.0,
        "low": px - 1.0,
        "close": px,
        vol_name: rng.integers(1_000, 5_000, periods),
    }, index=idx)
    return df


def test_resample_weekly_shape():
    out = resample_ohlcv(_mk(), "W")
    assert "date" in out.columns
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(out) > 0


def test_resample_ohlc_consistency():
    out = resample_ohlcv(_mk(), "W")
    assert (out["high"] >= out["low"]).all()
    assert (out["high"] >= out["close"]).all()
    assert (out["low"] <= out["open"]).all()


def test_resample_missing_columns_raises():
    df = _mk().drop(columns=["high"])
    with pytest.raises(ValueError):
        resample_ohlcv(df)


def test_resample_volume_alias_vol():
    out = resample_ohlcv(_mk(vol_name="vol"), "W")
    assert "volume" in out.columns
    assert "vol" not in out.columns


def test_resample_volume_alias_amount():
    out = resample_ohlcv(_mk(vol_name="amount"), "W")
    assert "volume" in out.columns


def test_resample_empty_safe():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = resample_ohlcv(empty)
    assert len(out) == 0
    assert "date" in out.columns


def test_resample_none_safe():
    out = resample_ohlcv(None)
    assert len(out) == 0


def test_resample_label_forwarded():
    # 同时传 label/closed 不应抛异常（成对传入避免 label/closed 告警）
    out = resample_ohlcv(_mk(), "W", label="left", closed="left")
    assert len(out) > 0


def test_resample_vwap_present():
    out = resample_vwap(_mk())
    assert "vwap" in out.columns
    assert (out["vwap"] > 0).all()
    # VWAP 应落在当周高低之间（近似）
    assert (out["vwap"] <= out["high"] + 1e-6).all()
    assert (out["vwap"] >= out["low"] - 1e-6).all()


def test_resample_vwap_missing_price_raises():
    with pytest.raises(ValueError):
        resample_vwap(_mk(), price="nope")


def test_resample_vwap_missing_volume_raises():
    df = _mk().drop(columns=["volume"])
    with pytest.raises(ValueError):
        resample_vwap(df)


def test_resample_vwap_empty_safe():
    out = resample_vwap(None)
    assert len(out) == 0
    assert "vwap" in out.columns
