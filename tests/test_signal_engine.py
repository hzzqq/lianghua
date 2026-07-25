"""signal 模块打磨：覆盖引擎、打分、排序、共识聚合与守卫。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.signal.engine import SignalEngine
from lianghua.signal.score import composite_score
from lianghua.signal.rank import rank_signal, rank_to_position


class _FakeStrat:
    """可调的假策略：返回固定信号序列。"""

    def __init__(self, sig):
        self._sig = sig

    def generate_signals(self, df):
        return self._sig


def _df():
    return pd.DataFrame({"close": [10, 11, 12, 13]}, index=pd.date_range("2024-01-01", periods=4))


def test_register_rejects_non_strategy():
    with pytest.raises(TypeError):
        SignalEngine().register("bad", 123)


def test_run_emits_nonzero_events():
    sig = pd.Series([0, 1, 0, -1], index=_df().index)
    eng = SignalEngine().register("a", _FakeStrat(sig))
    out = eng.run(_df())
    assert list(out["signal"]) == [1, -1]
    assert list(out["rule"]) == ["a", "a"]


def test_run_handles_2d_output():
    # 返回单列 DataFrame（implicit 2D 分支）
    sig = pd.DataFrame({"x": [0, 1, 0, -1]}, index=_df().index)
    eng = SignalEngine().register("a", _FakeStrat(sig))
    out = eng.run(_df())
    assert list(out["signal"]) == [1, -1]


def test_run_skips_nan():
    sig = pd.Series([0, np.nan, 1, -1], index=_df().index)
    eng = SignalEngine().register("a", _FakeStrat(sig))
    out = eng.run(_df())
    # NaN 被安全跳过，不应抛 int(nan)
    assert list(out["signal"]) == [1, -1]


def test_consensus_majority_and_min_votes():
    idx = _df().index
    s1 = pd.Series([1, 1, 0, -1], index=idx)
    s2 = pd.Series([1, 0, 0, -1], index=idx)
    eng = SignalEngine().register("r1", _FakeStrat(s1)).register("r2", _FakeStrat(s2))
    c = eng.consensus(_df())
    # 日期0: 两规则都 +1 → +1；日期3: 都 -1 → -1；其余非同向不足 min_votes(默认1 但需同向)
    assert c.loc[idx[0]] == 1
    assert c.loc[idx[3]] == -1
    # 日期1 仅一个 +1 → 达到 min_votes=1 → +1
    assert c.loc[idx[1]] == 1
    # 日期2 全 0 → 0
    assert c.loc[idx[2]] == 0

    # 提高门槛：日期1 只有一票 +1，min_votes=2 时应被抑制
    c2 = eng.consensus(_df(), min_votes=2)
    assert c2.loc[idx[1]] == 0
    assert c2.loc[idx[0]] == 1


def test_composite_score_constant_no_inf():
    # 两个常数信号不应产生 ±inf
    a = pd.Series([5.0] * 10)
    b = pd.Series([2.0] * 10)
    score = composite_score({"a": a, "b": b}, normalize=True)
    assert np.all(np.isfinite(score))
    assert score.abs().max() == 0.0


def test_composite_score_weighted():
    idx = pd.date_range("2024-01-01", periods=5)
    a = pd.Series([1.0, 2, 3, 4, 5], index=idx)
    b = pd.Series([1.0, 3, 2, 5, 4], index=idx)
    score = composite_score({"a": a, "b": b}, weights={"a": 1, "b": 1})
    assert np.all(np.isfinite(score))
    # 复算等权标准化求和，与函数实现一致
    za = (a - a.mean()) / a.std()
    zb = (b - b.mean()) / b.std()
    expected = ((za + zb) / 2).rename("score")
    pd.testing.assert_series_equal(score, expected, check_names=False)


def test_composite_score_empty():
    assert composite_score({}).empty


def test_rank_signal_empty():
    out = rank_signal({})
    assert out.empty


def test_rank_to_position_bounds():
    score = pd.Series([0.9, 0.5, 0.1])
    pos = rank_to_position(score, long_th=0.7, short_th=0.3)
    assert list(pos) == [1, 0, -1]
