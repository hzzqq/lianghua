"""Round 7：BacktestEngine 成本修正 + 资金约束 + stats() 可观测。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.backtest.engine import BacktestEngine, BacktestResult
from lianghua.risk.cost import STOCK_COST
from lianghua.core.assets import AssetType


def _df(prices, symbol="X"):
    idx = pd.date_range("2023-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"date": idx, "close": prices, "symbol": symbol})


def test_run_basic_produces_equity_and_trades():
    prices = [100, 101, 102, 103, 104, 103, 102, 101, 100, 99]
    sig = pd.Series([1] + [0] * 8 + [-1], index=pd.date_range("2023-01-01", periods=10, freq="D"))
    df = _df(prices)
    eng = BacktestEngine(init_cash=1_000_000)
    res = eng.run(df, sig)
    assert len(res.equity) == 10
    assert any(t["side"] == "BUY" for t in res.trades)
    assert any(t["side"] == "SELL" for t in res.trades)
    st = res.stats()
    assert set(st) >= {"total_return", "n_trades", "max_drawdown", "final_equity"}


def test_no_overspend_cash_never_negative():
    rng = np.random.default_rng(1)
    prices = 100 + np.cumsum(rng.normal(0, 2, 200))
    prices = np.maximum(prices, 1.0)
    sig = pd.Series((rng.normal(0, 1, 200) > 0.5).astype(int) * 2 - 1,
                    index=pd.date_range("2023-01-01", periods=200, freq="D"))
    df = _df(prices.tolist())
    eng = BacktestEngine(init_cash=100_000)
    res = eng.run(df, sig)
    for t in res.trades:
        assert t["cash_after"] >= -1e-6, t


def test_cost_model_used_not_double_slippage():
    prices = [100, 100, 100, 100, 100]
    sig = pd.Series([1, 0, 0, 0, -1], index=pd.date_range("2023-01-01", periods=5, freq="D"))
    df = _df(prices)
    eng = BacktestEngine(init_cash=1_000_000, cost=STOCK_COST)
    res = eng.run(df, sig)
    buys = [t for t in res.trades if t["side"] == "BUY"]
    assert buys, "应有买入成交"
    # fill 价应等于 价格*(1+滑点)，仅计一次滑点
    assert buys[0]["price"] == pytest.approx(100 * (1 + STOCK_COST.slippage), rel=1e-3)
    # 小单费用应受 min_fee 托底（>=5）
    assert buys[0]["fee"] >= STOCK_COST.min_fee - 1e-6


def test_stop_loss_triggers_sell():
    # 买入后价格下跌超过 10% 触发止损
    prices = [100, 100, 89, 89, 89]
    sig = pd.Series([1, 0, 0, 0, 0], index=pd.date_range("2023-01-01", periods=5, freq="D"))
    df = _df(prices)
    eng = BacktestEngine(init_cash=1_000_000)
    res = eng.run(df, sig)
    reasons = [t.get("reason") for t in res.trades if t["side"] == "SELL"]
    assert "stop_loss" in reasons


def test_no_zero_qty_trades():
    prices = [100, 101, 102, 103, 104]
    sig = pd.Series([1, 0, 0, 0, -1], index=pd.date_range("2023-01-01", periods=5, freq="D"))
    df = _df(prices)
    eng = BacktestEngine(init_cash=1_000_000)
    res = eng.run(df, sig)
    assert all(t["qty"] > 0 for t in res.trades)


def test_empty_df_safe():
    df = _df([])
    eng = BacktestEngine()
    res = eng.run(df, pd.Series(dtype=int))
    assert len(res.equity) == 0
    assert res.stats()["n_trades"] == 0


def test_to_frame_shape():
    prices = [100, 101, 102, 103, 104, 103, 102, 101, 100, 99]
    sig = pd.Series([1] + [0] * 8 + [-1], index=pd.date_range("2023-01-01", periods=10, freq="D"))
    df = _df(prices)
    res = BacktestEngine().run(df, sig)
    fr = res.to_frame()
    assert len(fr) == 10
    assert "equity" in fr.columns and "close" in fr.columns
