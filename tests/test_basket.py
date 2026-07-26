"""迭代 47：backtest/basket 打磨 —— 索引对齐修复 + 规格校验 + basket_daily_returns。"""
import numpy as np
import pandas as pd

from lianghua.backtest.basket import (
    validate_specs, normalize_weights, basket_daily_returns, basket_backtest,
)


def _make_gw():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    data = {
        "AAA": pd.DataFrame({"date": dates, "close": [10.0, 11.0, 12.0, 11.5, 13.0]}),
        "BBB": pd.DataFrame({"date": dates, "close": [20.0, 19.0, 21.0, 22.0, 21.5]}),
        # 首日出价为 0 的非法标的
        "ZERO": pd.DataFrame({"date": dates, "close": [0.0, 1.0, 2.0, 3.0, 4.0]}),
        # 含缺失交易日，用于触发 dropna 对齐
        "GAP": pd.DataFrame({"date": dates[1:], "close": [5.0, 6.0, 7.0, 8.0]}),
    }

    class FakeGW:
        def fetch(self, sym, start, end, asset=None):
            return data.get(sym)

    return FakeGW()


def test_normalize_weights():
    w = normalize_weights([{"symbol": "A", "weight": 1}, {"symbol": "B", "weight": 3}])
    assert abs(w["A"] - 0.25) < 1e-9 and abs(w["B"] - 0.75) < 1e-9


def test_validate_specs_rejects():
    import pytest
    with pytest.raises(ValueError):
        validate_specs([])
    with pytest.raises(TypeError):
        validate_specs("not a list")
    with pytest.raises(ValueError):
        validate_specs([{"symbol": "", "weight": 1}])
    with pytest.raises(ValueError):
        validate_specs([{"symbol": "A"}])  # 缺 weight
    with pytest.raises(ValueError):
        validate_specs([{"symbol": "A", "weight": "x"}])
    with pytest.raises(ValueError):
        validate_specs([{"symbol": "A", "weight": float("nan")}])


def test_basket_backtest_alignment():
    gw = _make_gw()
    specs = [{"symbol": "AAA", "weight": 0.5}, {"symbol": "BBB", "weight": 0.5}]
    res = basket_backtest(specs, "2020-01-01", "2020-01-05", gw=gw)
    eq = res["equity"]
    # 无 NaN 行（dropna 后索引已对齐）
    assert eq.notna().all()
    # daily_returns 与 equity 索引一致
    assert res["daily_returns"].index.equals(eq.index)
    # 首日净值 = init_cash
    assert abs(eq.iloc[0] - 1_000_000) < 1e-6
    # 权重和为 1
    assert abs(sum(res["weights"].values()) - 1.0) < 1e-9


def test_basket_daily_returns_matches():
    gw = _make_gw()
    specs = [{"symbol": "AAA", "weight": 0.5}, {"symbol": "BBB", "weight": 0.5}]
    dr = basket_daily_returns(specs, "2020-01-01", "2020-01-05", gw=gw)
    res = basket_backtest(specs, "2020-01-01", "2020-01-05", gw=gw)
    assert dr.equals(res["daily_returns"])


def test_basket_zero_first_price_raises():
    import pytest
    gw = _make_gw()
    with pytest.raises(ValueError):
        basket_backtest([{"symbol": "ZERO", "weight": 1}], "2020-01-01", "2020-01-05", gw=gw)


def test_basket_missing_data_raises():
    import pytest
    gw = _make_gw()
    with pytest.raises(ValueError):
        basket_backtest([{"symbol": "NOPE", "weight": 1}], "2020-01-01", "2020-01-05", gw=gw)
