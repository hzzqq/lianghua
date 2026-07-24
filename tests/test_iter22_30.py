"""迭代 22–30 综合集成测试（离线，桩网关避免联网）。

覆盖：VaR/CVaR、滚动绩效、篮子回测、期权组合、期货跨期套利、
基金筛选、向量化回测、数据 IO、信号推送。
运行：python tests/test_iter22_30.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lianghua.data.gateway import DataGateway
from lianghua.backtest.basket import basket_backtest
from lianghua.risk.var import var_report
from lianghua.perf.rolling import rolling_report
from lianghua.option.strategy import butterfly, iron_condor, payoff_curve, combo_payoff
from lianghua.backtest.future_spread import FutureSpread, spread_series
from lianghua.fund.screener import FundScreener
from lianghua.backtest.vectorized import vectorized_backtest
from lianghua import io as D
from lianghua.execution.notify import SignalNotifier, LogChannel


class _StubGW(DataGateway):
    def fetch(self, symbol, start, end, freq="daily", asset=None):
        a = asset or __import__("lianghua.core.assets", fromlist=["detect_asset_type"]).detect_asset_type(symbol)
        return self._demo_data(symbol, start, end, a)


def _gw():
    return _StubGW()


def test_var_cvar():
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.0005, 0.015, 400))
    rep = var_report(ret, notional=1_000_000)
    assert 0.95 in rep["置信度"].values
    assert (rep["历史VaR"] > 0).all()


def test_rolling():
    rng = np.random.default_rng(2)
    eq = pd.Series((1 + rng.normal(0.0005, 0.015, 400)).cumprod() * 1_000_000)
    rr = rolling_report(eq, 60).dropna()
    assert len(rr) > 100


def test_basket():
    r = basket_backtest([{"symbol": "600519.SH", "weight": 0.5},
                         {"symbol": "510300.SH", "weight": 0.5}],
                        "2023-01-01", "2023-12-31", gw=_gw())
    assert "equity" in r and float(r["equity"].iloc[-1]) > 0
    assert abs(sum(r["weights"].values()) - 1.0) < 1e-9


def test_option_combos():
    bf = butterfly("CALL", 90, 100, 110, 1.0, 4.0, 1.0)
    assert combo_payoff(bf, 100) > combo_payoff(bf, 90)
    ic = iron_condor(90, 95, 105, 110, 0.8, 2.0, 2.0, 0.8)
    assert payoff_curve(ic, 80, 120)["pnl"].max() > 0


def test_future_spread():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    n = len(dates)
    base = rng.normal(0, 0.01, n).cumsum()
    near = 3500 + base * 10
    far = near + 50 + rng.normal(0, 2, n)
    near_df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": near})
    far_df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": far})
    r = FutureSpread(lots=2, multiplier=10).run(near_df, far_df)
    assert "equity" in r and len(r["signals"]) == n


def test_fund_screener():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2022-01-01", "2023-12-31")
    funds = {f"F{i}": (1 + rng.normal(0.0006, 0.012, len(dates))).cumprod()
             for i in range(4)}
    res = FundScreener.from_frame(pd.DataFrame(funds, index=dates)).screen()
    assert len(res) == 4 and "综合分" in res.columns


def test_vectorized():
    rng = np.random.default_rng(5)
    close = 100 * (1 + rng.normal(0.0003, 0.015, 1000)).cumprod()
    df = pd.DataFrame({"date": pd.bdate_range("2000-01-01", periods=1000), "close": close})
    sma = pd.Series(close).rolling(20).mean()
    sig = pd.Series(np.where(sma > pd.Series(close).rolling(60).mean(), 1, 0), index=df.index)
    r = vectorized_backtest(df, sig, cost_rate=0.0005)
    assert float(r["equity"].iloc[-1]) > 0


def test_io():
    t = tempfile.mkdtemp()
    df = pd.DataFrame({"date": ["2023-01-02"], "open": [10], "high": [11],
                       "low": [9.8], "close": [10.6], "volume": [1000]})
    bp = os.path.join(t, "bars.csv")
    D.write_bars(df, bp)
    assert D.read_bars(bp).iloc[0]["close"] == 10.6
    tp = os.path.join(t, "trades.csv")
    D.write_trades([{"date": "2023-01-02", "side": "BUY", "price": 10.6}], tp)
    assert D.read_trades(tp)[0]["side"] == "BUY"
    if D.excel_available():
        ep = os.path.join(t, "rep.xlsx")
        D.to_excel({"bars": df}, ep)
        assert os.path.exists(ep)


def test_notify():
    t = tempfile.mkdtemp()
    n = SignalNotifier(channels=[LogChannel(os.path.join(t, "s.log"))])
    r = n.notify_signal("600519.SH", "BUY", price=1700.0, strategy="sma_cross")
    assert r["delivered"] >= 1 and os.path.getsize(os.path.join(t, "s.log")) > 0


def test_notify_robust_against_preset_logging():
    """回归：即便 logging 已被其它模块预配置（root 已有 handler），
    LogChannel 仍必须独立把信号写入自己的文件。"""
    import logging as _logging
    _logging.basicConfig(filename=os.devnull)  # 模拟 root 已被占用
    t = tempfile.mkdtemp()
    f = os.path.join(t, "robust.log")
    ch = LogChannel(f)
    assert ch.path() == f
    r = ch.send({"type": "signal", "symbol": "000001.SZ", "action": "SELL"})
    ch.close()
    assert r["ok"] is True
    assert os.path.exists(f) and os.path.getsize(f) > 0
    # 验证确实写到了自己的文件，而非 devnull
    with open(f, encoding="utf-8") as fh:
        assert "SELL" in fh.read()


def test_logchannel_close_idempotent():
    t = tempfile.mkdtemp()
    ch = LogChannel(os.path.join(t, "c.log"))
    ch.send({"a": 1})
    ch.close()
    ch.close()  # 重复关闭不应抛错
    assert os.path.exists(os.path.join(t, "c.log"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} 通过")
    raise SystemExit(0 if passed == len(fns) else 1)
