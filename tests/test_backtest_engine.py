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
    # 末位留 0：默认 execution_lag=1 会把每个信号推迟一根 K 线执行，
    # 因此最后 lag 个信号会被推到回测区间之外而不成交（这是正确语义，
    # 见 test_trailing_signal_deferred_out_of_range）。要让卖出指令在区间内
    # 真正执行，信号序列末尾必须留出 lag 根 K 线。
    sig = pd.Series([1] + [0] * 7 + [-1, 0],
                    index=pd.date_range("2023-01-01", periods=10, freq="D"))
    df = _df(prices)
    eng = BacktestEngine(init_cash=1_000_000)
    res = eng.run(df, sig)
    assert len(res.equity) == 10
    assert any(t["side"] == "BUY" for t in res.trades)
    assert any(t["side"] == "SELL" for t in res.trades)
    st = res.stats()
    assert set(st) >= {"total_return", "n_trades", "max_drawdown", "final_equity"}


def test_trailing_signal_deferred_out_of_range():
    """延迟执行的固有语义：末尾 lag 个信号不会在区间内成交。

    这不是缺陷——信号产生于 t 日收盘，只能在 t+1 及之后执行；若回测
    到 t 日为止，t 日产生的信号就落在区间之外。反过来若强行执行它，
    就等于假设能在信号产生的同一刻成交，即重新引入前视偏差。
    """
    prices = [100, 101, 102, 103, 104]
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    sig = pd.Series([0, 0, 0, 0, 1], index=idx)  # 最后一根才出买入信号
    df = _df(prices)
    res = BacktestEngine().run(df, sig)
    assert res.trades == [], "末位信号应被延迟到区间外，不得在区间内成交"
    # 对照：lag=0 时同一信号会在区间内成交（研究用途，非实盘假设）
    res0 = BacktestEngine(execution_lag=0).run(df, sig)
    assert any(t["side"] == "BUY" for t in res0.trades)


def test_last_signal_lost_warning_is_observable():
    """末尾 N 个信号被延迟出区间时，结果里必须可查证，不能静默。"""
    prices = [100, 101, 102, 103, 104]
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    sig = pd.Series([1, 0, 0, 0, -1], index=idx)
    res = BacktestEngine().run(_df(prices), sig)
    # exec_signals 暴露实际下单用的信号，用户可比对自己给的 signals
    assert res.exec_signals is not None
    assert float(res.exec_signals.iloc[-1]) == 0.0
    assert res.stats()["execution_lag"] == 1


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
