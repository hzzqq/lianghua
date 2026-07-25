"""risk/drawdown 单元测试：守卫 + Top-N 回撤明细表。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.risk.drawdown import (
    drawdown_series,
    drawdown_table,
    max_drawdown_info,
)


def _eq(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_drawdown_series_basic():
    eq = _eq([100, 90, 80, 85, 100])
    dd = drawdown_series(eq)
    assert dd.iloc[2] == pytest.approx(-0.2)            # 80 vs peak 100
    assert dd.iloc[4] == pytest.approx(0.0)             # 收复，回到峰值
    assert (dd <= 0).all()


def test_drawdown_series_rejects_nan():
    with pytest.raises(ValueError):
        drawdown_series(_eq([100, np.nan, 80]))


def test_drawdown_series_rejects_nonpositive():
    with pytest.raises(ValueError):
        drawdown_series(_eq([100, 0, 80]))


def test_drawdown_series_empty():
    with pytest.raises(ValueError):
        drawdown_series(pd.Series([], dtype=float))


def test_max_drawdown_info_no_recovery():
    # 一路下跌，从未收复
    eq = _eq([100, 90, 80, 70])
    info = max_drawdown_info(eq)
    assert info["max_drawdown"] == pytest.approx(-0.3)
    assert info["recovery_date"] is None
    assert info["duration_days"] == 3                  # 实际已持续天数，而非 0


def test_max_drawdown_info_monotonic_up():
    eq = _eq([100, 105, 110, 120])
    info = max_drawdown_info(eq)
    assert info["max_drawdown"] == pytest.approx(0.0)
    assert info["recovery_date"] == eq.index[-1]


def test_max_drawdown_single_point():
    info = max_drawdown_info(_eq([100]))
    assert info["max_drawdown"] == 0.0


def test_drawdown_table_top_n_and_order():
    # 构造两段回撤：深 -0.3，浅 -0.1
    eq = _eq([100, 70, 100, 95, 90, 100])
    tbl = drawdown_table(eq, top_n=5)
    assert len(tbl) == 2
    assert tbl["depth"].iloc[0] == pytest.approx(-0.3)  # 深者在前
    assert tbl["depth"].iloc[1] == pytest.approx(-0.1)
    assert tbl["recovery_gain"].iloc[0] == pytest.approx(1 / 0.7 - 1)  # 70->100


def test_drawdown_table_unrecovered_tail():
    eq = _eq([100, 90, 80, 75])                          # 末段未收复
    tbl = drawdown_table(eq, top_n=5)
    assert len(tbl) == 1
    assert tbl["recovery_date"].iloc[0] is None
    assert np.isnan(tbl["recovery_gain"].iloc[0])


def test_drawdown_table_top_n_limits():
    eq = _eq([100, 90, 100, 80, 100, 70, 100])
    tbl = drawdown_table(eq, top_n=2)
    assert len(tbl) == 2


def test_drawdown_table_invalid_top_n():
    with pytest.raises(ValueError):
        drawdown_table(_eq([100, 90, 100]), top_n=0)
