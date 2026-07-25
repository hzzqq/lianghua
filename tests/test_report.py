"""report 模块打磨测试：HTML/tearsheet 对非有限与空输入的鲁棒性 + 导出清洗。"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from lianghua.report import (
    export_backtest,
    render_backtest_html,
    sanitize_metrics,
    tearsheet,
)


class _Result:
    def __init__(self, equity, trades):
        self.equity = equity
        self.trades = trades


def _ok_equity():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.Series(np.linspace(1.0, 1.3, 30), index=idx)


def test_render_html_normal():
    eq = _ok_equity()
    trades = [
        {"date": "2024-01-05", "side": "BUY", "price": 1.05, "qty": 100},
        {"date": "2024-01-20", "side": "SELL", "price": 1.20, "qty": 100},
    ]
    html = render_backtest_html(_Result(eq, trades), "BTC", "ma_cross")
    assert "<!doctype html>" in html
    assert "夏普" in html
    assert "1.05" in html  # 交易价格保留


def test_render_html_nan_equity_no_crash():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    eq = pd.Series([1.0, np.nan, 2.0, np.inf, 1.5, np.nan, 1.2, 0.9, 1.1, 1.3], index=idx)
    html = render_backtest_html(_Result(eq, []), "X", "s")
    assert "<!doctype html>" in html
    assert "NaN" not in html  # 卡片不应出现裸 NaN
    assert "nan" not in html


def test_render_html_none_trades_no_crash():
    html = render_backtest_html(_Result(_ok_equity(), None), "X", "s")
    assert "<!doctype html>" in html
    assert "前 0 笔" in html


def test_render_html_short_equity_placeholder():
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    html = render_backtest_html(_Result(pd.Series([1.0], index=idx), []), "X", "s")
    assert "无有效权益数据" in html


def test_tearsheet_normal():
    df = tearsheet(_ok_equity())
    assert set(df.columns) == {"指标", "数值"}
    assert len(df) == 8
    assert df["数值"].notna().any()


def test_tearsheet_empty():
    df = tearsheet(pd.Series([], dtype=float))
    assert len(df) == 0


def test_tearsheet_zero_first_value():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    eq = pd.Series([0.0, 1.0, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 1.4, 1.6], index=idx)
    df = tearsheet(eq)
    assert df["数值"].notna().any()  # 不应整体崩溃，且累计收益为 NaN 而非 inf


def test_sanitize_metrics_cleans_nonfinite_and_strings():
    raw = {
        "a": float("nan"),
        "b": float("inf"),
        "c": None,
        "d": "12.5",
        "e": 3,
        "nested": {"x": float("nan"), "y": 1.0},
        "lst": [float("inf"), 2.0],
    }
    out = sanitize_metrics(raw)
    assert out["a"] is None
    assert out["b"] is None
    assert out["c"] is None
    assert out["d"] == 12.5
    assert out["e"] == 3.0
    assert out["nested"]["x"] is None
    assert out["nested"]["y"] == 1.0
    assert out["lst"][0] is None
    assert out["lst"][1] == 2.0
    # 必须是合法 JSON
    json.dumps(out)


def test_export_backtest_writes_valid_json_and_trades():
    eq = _ok_equity()
    metrics = {
        "sharpe": 1.2,
        "total_return": float("nan"),
        "max_drawdown": -0.1,
    }
    trades = [{"date": "2024-01-05", "side": "BUY", "price": 1.05, "qty": 100}]
    with tempfile.TemporaryDirectory() as d:
        export_backtest(
            {"equity": eq, "metrics": metrics, "trades": trades}, d, name="bt"
        )
        assert os.path.exists(os.path.join(d, "bt_equity.csv"))
        assert os.path.exists(os.path.join(d, "bt_trades.csv"))
        with open(os.path.join(d, "bt_metrics.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert data["sharpe"] == 1.2
        assert data["total_return"] is None
        assert data["max_drawdown"] == -0.1
