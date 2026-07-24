"""pandas 4.0 前向兼容回归测试。

锁定 .sum(1) 等位置参数轴调用已改为关键字轴（mat.sum(axis=1)），
避免 pandas 4.0 升级后这些调用直接报错。
"""
import warnings

import numpy as np
import pandas as pd

from lianghua.signal.meta import majority_vote, weighted_meta
from lianghua.perf.contrib import time_contrib


def _no_pandas4_warning(func):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ret = func()
    for w in rec:
        msg = str(w.message)
        assert "keyword-only" not in msg, f"出现 pandas 4.0 弃用警告: {msg}"
    return ret


def test_majority_vote_basic():
    out = _no_pandas4_warning(
        lambda: majority_vote({"a": [1, -1, 0], "b": [1, 1, -1], "c": [-1, 1, 1]})
    )
    assert list(out.values) == [1, 1, 0]  # 票数为 1,1,0（平票取 0）
    assert out.name == "meta"


def test_majority_vote_empty():
    out = _no_pandas4_warning(lambda: majority_vote({}))
    assert len(out) == 0


def test_weighted_meta_basic():
    out = _no_pandas4_warning(
        lambda: weighted_meta({"a": [1.0, -1.0], "b": [1.0, 1.0]}, weights={"a": 1, "b": 3})
    )
    # 加权后 (1*1+1*3)=4>0 -> 1; (-1*1+1*3)=2>0 -> 1
    assert list(out.values) == [1.0, 1.0]


def test_time_contrib_axis():
    w = pd.Series([0.5, 0.5], index=["x", "y"])
    ret = pd.DataFrame({"x": [0.01, 0.02], "y": [0.03, 0.04]})
    out = _no_pandas4_warning(lambda: time_contrib(w, ret))
    assert list(out.values) == [0.02, 0.03]
    assert out.name == "contrib"
