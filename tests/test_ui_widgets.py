"""ui/widgets 打磨测试：空态/错误态降级、安全格式化。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lianghua.ui.widgets import (
    drawdown_fig,
    empty_figure,
    equity_fig,
    safe_num,
    safe_pct,
)


def _ok():
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    return pd.Series(np.linspace(1.0, 1.3, 20), index=idx)


def test_safe_pct():
    assert safe_pct(0.123) == "12.3%"
    assert safe_pct(float("nan")) == "N/A"
    assert safe_pct(float("inf")) == "N/A"


def test_safe_num():
    assert safe_num(1.23456) == "1.23"
    assert safe_num(None) == "N/A"
    assert safe_num(float("nan")) == "N/A"


def test_empty_figure_has_no_traces():
    fig = empty_figure("无数据")
    assert len(fig.data) == 0


def test_equity_fig_normal_has_trace():
    fig = equity_fig(_ok())
    assert len(fig.data) == 1


def test_equity_fig_empty_degrades():
    fig = equity_fig(pd.Series([], dtype=float))
    assert len(fig.data) == 0  # 占位图，无数据线


def test_equity_fig_all_nan_degrades():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    fig = equity_fig(pd.Series([np.nan, np.inf, np.nan], index=idx[:3]))
    assert len(fig.data) == 0


def test_drawdown_fig_normal_has_trace():
    fig = drawdown_fig(_ok())
    assert len(fig.data) == 1


def test_drawdown_fig_empty_degrades():
    fig = drawdown_fig(pd.Series([], dtype=float))
    assert len(fig.data) == 0
