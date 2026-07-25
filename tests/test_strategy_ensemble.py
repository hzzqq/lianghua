"""strategy/ensemble 单元测试：组合方式 + 守卫 + 索引对齐。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.strategy.ensemble import Ensemble


def _df(n=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": np.arange(1, n + 1, dtype=float)}, index=idx)


class FakeStrategy:
    def __init__(self, vals):
        self.vals = vals

    def generate_signals(self, df):
        return pd.Series(self.vals, index=df.index)


class OffsetStrategy:
    """故意返回与 df 不同索引的信号，用于测试对齐守卫。"""

    def generate_signals(self, df):
        return pd.Series([1, 1, 1], index=pd.date_range("2024-02-01", periods=3))


def test_weighted_balanced_is_zero():
    df = _df()
    e = (Ensemble().add("a", FakeStrategy([1] * 10), 1.0)
         .add("b", FakeStrategy([-1] * 10), 1.0))
    sig = e.signals(df, threshold=0.1, method="weighted")
    assert (sig == 0).all()


def test_weighted_positive_signal():
    df = _df()
    e = (Ensemble().add("a", FakeStrategy([1] * 10), 1.0)
         .add("b", FakeStrategy([0.5] * 10), 1.0))
    sig = e.signals(df, 0.1)
    assert (sig == 1).all()


def test_add_signal_direct():
    df = _df(5)
    e = Ensemble().add_signal("x", pd.Series([1, 1, 1, 1, 1], index=df.index), 1.0)
    sig = e.signals(df, 0.1)
    assert (sig == 1).all()


def test_vote_method():
    df = _df()
    # 多空等权投票抵消 -> 0
    e = (Ensemble().add("a", FakeStrategy([2] * 10), 1.0)
         .add("b", FakeStrategy([-3] * 10), 1.0))
    assert (e.signals(df, 0.1, method="vote") == 0).all()
    # 多头权重更高 -> +1
    e2 = (Ensemble().add("a", FakeStrategy([2] * 10), 2.0)
          .add("b", FakeStrategy([-3] * 10), 1.0))
    assert (e2.signals(df, 0.1, method="vote") == 1).all()


def test_zscore_method_has_spread():
    df = _df()
    e = Ensemble().add("a", FakeStrategy(list(np.linspace(-1, 1, 10))), 1.0)
    raw = e.raw(df, "zscore")
    assert raw.std() > 0
    assert not raw.isna().any()


def test_nonfinite_weight_rejected():
    with pytest.raises(ValueError):
        Ensemble().add("a", FakeStrategy([1] * 10), float("nan"))


def test_nan_signal_rejected():
    with pytest.raises(ValueError):
        Ensemble().add_signal("x", pd.Series([1, np.nan, 1]), 1.0)


def test_empty_ensemble_raises():
    with pytest.raises(ValueError):
        Ensemble().signals(_df())


def test_nan_threshold_rejected():
    e = Ensemble().add("a", FakeStrategy([1] * 10), 1.0)
    with pytest.raises(ValueError):
        e.signals(_df(), float("nan"))


def test_index_misalignment_safe():
    df = _df()
    e = Ensemble().add("a", OffsetStrategy(), 1.0)
    sig = e.signals(df, 0.1)
    assert not sig.isna().any()           # 重对齐填 0，无错位 NaN


def test_weights_normalized():
    e = (Ensemble().add("a", FakeStrategy([1] * 10), 1.0)
         .add("b", FakeStrategy([1] * 10), 3.0))
    assert e.weights() == {"a": 0.25, "b": 0.75}


def test_invalid_method_rejected():
    e = Ensemble().add("a", FakeStrategy([1] * 10), 1.0)
    with pytest.raises(ValueError):
        e.raw(_df(), "bogus")
