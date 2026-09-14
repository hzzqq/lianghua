"""registry.get_strategy 必须把 **kwargs 转发到底层策略函数（FnStrategy）。

回归：旧实现 _FN_REGISTRY 的 lambda 丢弃 **kwargs，导致
get_strategy("bollinger", period=15) 直接抛 TypeError，参数敏感性网格无法运行。
"""
import numpy as np
import pandas as pd

from lianghua.strategy.registry import get_strategy


def _df(n=120, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": close, "open": close, "high": close,
                         "low": close, "volume": 1.0}, index=idx)


def test_period_forwards_to_bollinger():
    df = _df()
    s15 = get_strategy("bollinger", period=15).generate_signals(df)
    s20 = get_strategy("bollinger", period=20).generate_signals(df)
    # 不同 period 应产生不同信号（证明 kwarg 真的进了底层函数）
    assert not s15.equals(s20)
    assert set(s20.unique()).issubset({-1, 0, 1})


def test_window_forwards_to_ou():
    df = _df()
    # ou 的主旋钮是 window（非 period）
    s = get_strategy("ou", window=40).generate_signals(df)
    assert set(np.unique(s.values)).issubset({-1, 0, 1})


def test_default_call_still_works():
    # 不传 kwargs 也应正常（向后兼容）
    df = _df()
    s = get_strategy("rsi").generate_signals(df)
    assert set(np.unique(s.values)).issubset({-1, 0, 1})


def test_unknown_kwarg_rejected_cleanly():
    df = _df()
    # 底层函数不接受的 kwarg 应抛 TypeError（而非静默吞掉）
    try:
        get_strategy("bollinger", not_a_real_param=1).generate_signals(df)
    except TypeError:
        pass
    else:
        raise AssertionError("不支持的 kwarg 应被底层函数拒绝")
