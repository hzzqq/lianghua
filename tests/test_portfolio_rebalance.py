"""Round 5：portfolio.rebalance 输入守卫 + 漂移带再平衡 + 回撤止损事件。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.portfolio.rebalance import rebalance, drawdown_stop


def _make(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    # 两个标的，起点为正，保证 base>0
    a = 100 + np.cumsum(rng.normal(0, 1, n))
    b = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_rebalance_shape_and_start():
    comp = _make()
    port = rebalance(comp, {"A": 0.6, "B": 0.4}, period=20)
    assert len(port) == len(comp)
    assert port.iloc[0] == pytest.approx(1_000_000.0)


def test_rebalance_normalizes_weights():
    comp = _make()
    port = rebalance(comp, {"A": 3.0, "B": 1.0}, period=20)
    # 权重自动归一化，期初权益仍为 init_cash
    assert port.iloc[0] == pytest.approx(1_000_000.0)


def test_drift_band_triggers_earlier_rebalance():
    comp = _make()
    # 制造 A 持续单边，使权重漂移
    comp = comp.copy()
    comp["A"] = comp["A"] * np.linspace(1, 3, len(comp))
    port_period, ev_p = rebalance(comp, {"A": 0.5, "B": 0.5}, period=1000, return_events=True)
    port_drift, ev_d = rebalance(comp, {"A": 0.5, "B": 0.5}, period=1000, drift_band=0.05, return_events=True)
    # 漂移带模式应在周期之外产生更多再平衡事件
    assert len(ev_d) > len(ev_p)
    assert len(ev_d) >= 1


def test_rebalance_events_on_period():
    comp = _make(n=60)
    _, ev = rebalance(comp, {"A": 0.5, "B": 0.5}, period=20, return_events=True)
    assert 20 in ev and 40 in ev


def test_rejects_zero_base():
    comp = _make()
    comp.iloc[0, 0] = 0.0
    with pytest.raises(ValueError):
        rebalance(comp, {"A": 0.5, "B": 0.5})


def test_rejects_negative_base():
    comp = _make()
    comp.iloc[0, 1] = -5.0
    with pytest.raises(ValueError):
        rebalance(comp, {"A": 0.5, "B": 0.5})


def test_rejects_unknown_ticker():
    comp = _make()
    with pytest.raises(ValueError):
        rebalance(comp, {"A": 0.5, "C": 0.5})


def test_rejects_empty_components():
    with pytest.raises(ValueError):
        rebalance(pd.DataFrame(), {"A": 1.0})


def test_rejects_all_zero_target():
    comp = _make()
    with pytest.raises(ValueError):
        rebalance(comp, {"A": 0.0, "B": 0.0})


def test_rejects_bad_drift_band():
    comp = _make()
    with pytest.raises(ValueError):
        rebalance(comp, {"A": 0.5, "B": 0.5}, drift_band=1.5)


def test_drawdown_stop_freezes_and_events():
    eq = pd.Series([100, 100, 90, 70, 70, 80, 90])
    out, ev = drawdown_stop(eq, max_dd=0.20, return_events=True)
    # 70/100-1 = -0.30 < -0.20 触发冻结
    assert ev == [3]
    # 冻结后权益保持为触发点的值
    assert out.iloc[3] == pytest.approx(70.0)
    assert out.iloc[4] == pytest.approx(70.0)
    assert out.iloc[5] == pytest.approx(70.0)


def test_drawdown_stop_no_trigger():
    eq = pd.Series([100, 100, 95, 92, 95])
    out = drawdown_stop(eq, max_dd=0.20)
    assert out.tolist() == eq.tolist()


def test_drawdown_stop_empty():
    out, ev = drawdown_stop(pd.Series(dtype=float), return_events=True)
    assert len(out) == 0 and ev == []
