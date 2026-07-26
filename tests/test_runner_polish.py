"""Cycle 53：backtest.runner 打磨——策略预检/空数据守卫 + validate_backtest_request。"""
import pandas as pd
import pytest
from unittest.mock import patch

from lianghua.backtest import runner
from lianghua.core.assets import AssetType


def test_validate_backtest_request_ok_and_bad():
    ok = runner.validate_backtest_request("600519.SH", "sma_cross")
    assert ok["asset_type"] == "stock"
    assert ok["ok"] is True
    bad = runner.validate_backtest_request("600519.SH", "no_such_strat")
    assert bad["ok"] is False and "no_such_strat" in bad["supported"] or "no_such_strat" == bad["strategy"]


def test_list_supported_strategies():
    assert isinstance(runner.list_supported_strategies("future"), list)
    assert "sma_cross" in runner.list_supported_strategies("stock")
    assert isinstance(runner.list_supported_strategies(None), dict)


def test_resolve_asset_failure_clear_error():
    with pytest.raises(ValueError):
        runner._resolve_asset("600519.SH", "not_a_valid_asset_type")


class _FakeGW:
    def __init__(self, df):
        self.df = df
        self.calls = 0

    def fetch(self, symbol, start, end, asset=None):
        self.calls += 1
        return self.df


class _FakeStrategy:
    def generate_signals(self, df):
        return pd.Series(1, index=df.index)


class _FakeResult:
    pass


def test_run_backtest_unknown_strategy_before_fetch():
    # 隐性修复：非法策略应在取数前校验，gw.fetch 不应被调用
    gw = _FakeGW(pd.DataFrame({"close": [1, 2, 3]}))
    with patch.object(runner, "get_strategy", side_effect=AssertionError("不应调用")):
        with pytest.raises(ValueError):
            runner.run_backtest("600519.SH", "2023-01-01", "2023-12-31",
                                strategy="nope", gw=gw)
    assert gw.calls == 0


def test_run_backtest_empty_data_raises():
    gw = _FakeGW(pd.DataFrame())  # 空数据
    with patch.object(runner, "get_strategy", side_effect=AssertionError("不应调用")):
        with pytest.raises(ValueError):
            runner.run_backtest("600519.SH", "2023-01-01", "2023-12-31",
                                strategy="sma_cross", gw=gw)
    assert gw.calls == 1  # 取数已发生，但被空数据守卫拦截


def test_run_backtest_happy_path_routes_stock_engine():
    gw = _FakeGW(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))
    with patch.object(runner, "get_strategy", return_value=_FakeStrategy()), \
         patch.object(runner, "BacktestEngine") as FakeEngine:
        FakeEngine.return_value.run.return_value = _FakeResult()
        res = runner.run_backtest("600519.SH", "2023-01-01", "2023-12-31",
                                  strategy="sma_cross", gw=gw)
        assert isinstance(res, _FakeResult)
        FakeEngine.return_value.run.assert_called_once()
        assert gw.calls == 1
