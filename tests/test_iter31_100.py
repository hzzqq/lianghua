"""迭代 31-100 综合集成测试（纯 pandas/numpy，离线桩数据）。

覆盖：策略(31-38,61-72)、指标、风险、绩效、组合、因子、信号、
执行、数据、报告、UI 组件、常量与版本清单。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

np.random.seed(0)

# ---- 公共演示数据 ----
N = 600
DATE = pd.bdate_range("2020-01-01", periods=N)
CLOSE = pd.Series(100 * (1 + np.random.RandomState(0).normal(0, 0.014, N)).cumprod(), index=DATE)
HIGH = (CLOSE * (1 + abs(np.random.RandomState(1).normal(0, 0.005, N))))
LOW = (CLOSE * (1 - abs(np.random.RandomState(2).normal(0, 0.005, N))))
X = pd.Series(100 * (1 + np.random.RandomState(3).normal(0, 0.014, N)).cumprod(), index=DATE)
RET = CLOSE.pct_change().dropna()
DFR = pd.DataFrame(np.random.RandomState(9).normal(0.0005, 0.01, (N, 4)),
                   columns=["A", "B", "C", "D"])
RSETS = pd.DataFrame(np.random.RandomState(4).normal(0.0005, 0.01, (N, 4)), columns=["A", "B", "C", "D"])


def ok(name, cond):
    assert cond, f"{name} 校验失败"
    print(f"  ✓ {name}")


# ===== 迭代 31-38：策略 + 指标 =====
def test_iter_31_38():
    from lianghua.strategy.turtle import turtle_signals
    from lianghua.strategy.rsi_strategy import rsi_signal
    from lianghua.strategy.bollinger import bollinger_signal
    from lianghua.strategy.grid import grid_signal
    from lianghua.strategy.mean_reversion_z import zscore_signal
    from lianghua.strategy.rotation import rotation_weights
    from lianghua.strategy.vol_target import vol_target_weights
    from lianghua.indicators import rsi, macd, kdj, boll, atr, cci, obv, vwap
    df = pd.DataFrame({"date": DATE, "open": CLOSE, "high": HIGH, "low": LOW,
                      "close": CLOSE, "volume": 1e5})
    ok("turtle", set(turtle_signals(df).unique()).issubset({-1, 0, 1}))
    ok("rsi_signal", (rsi_signal(df) != 0).any())
    ok("bollinger", (bollinger_signal(df) != 0).any())
    g = grid_signal(df)
    ok("grid", g.iloc[-1] in range(-5, 6) and g.dtype.kind in "iu")
    ok("zscore", (zscore_signal(df) != 0).any())
    ok("rotation", abs(rotation_weights(DFR, top=1).iloc[-1].sum() - 1) < 1e-6)
    ok("vol_target", abs(vol_target_weights(DFR).iloc[-1].sum() - 1) < 1e-6)
    ok("indicators", len(rsi(CLOSE)) == N and len(atr(df)) == N
       and len(cci(df)) == N and len(obv(df)) == N and len(vwap(df)) == N)
    ok("macd_kdj_boll", len(macd(CLOSE).dif) == N and len(kdj(df).k) == N and len(boll(CLOSE).mid) == N)


# ===== 迭代 39-50：风险/绩效/组合 =====
def test_iter_39_50():
    from lianghua.risk.budget import equal_risk_contribution, target_risk_budget
    from lianghua.risk.stress import stress_test
    from lianghua.risk.liquidity import amihud, liquidity_cost
    from lianghua.perf.attribution import brinson_attribution
    from lianghua.perf.trade_stats import trade_stats
    from lianghua.perf.monthly import monthly_table
    from lianghua.perf.benchmark import information_ratio
    from lianghua.portfolio.min_variance import min_variance
    from lianghua.portfolio.max_diversification import max_diversification
    from lianghua.portfolio.hrp import hrp
    from lianghua.portfolio.black_litterman import black_litterman
    A = np.random.RandomState(2).randn(4, 4)
    cov = A @ A.T / 4 + np.eye(4) * 0.001
    ok("ERC", abs(equal_risk_contribution(cov).sum() - 1) < 1e-6)
    ok("目标预算", abs(target_risk_budget(cov, [0.4, 0.3, 0.2, 0.1]).sum() - 1) < 1e-6)
    ok("压力测试", len(stress_test({"股票": 0.5}, {"加息": {"股票": -0.1}})) == 1)
    ok("amihud", 0 <= amihud(RET, pd.Series(1e6, RET.index), CLOSE) <= 1)
    ok("流动性成本", len(liquidity_cost({"股票": 0.5}, {"股票": 1e8})) == 1)
    idx = ["股票", "债券", "黄金", "现金"]
    pw = pd.Series([0.4, 0.3, 0.2, 0.1], idx)
    bw = pd.Series([0.3, 0.4, 0.2, 0.1], idx)
    rp = pd.Series([0.05, 0.02, 0.04, 0.01], idx)
    ok("Brinson", brinson_attribution(pw, bw, rp) is not None)
    ok("交易统计", trade_stats([{"side": "BUY", "pnl": None},
                                {"side": "SELL", "pnl": 100}])["num_trades"] == 2)
    ok("月度表", monthly_table(CLOSE).shape[1] == 12)
    ok("IR", abs(information_ratio(CLOSE, CLOSE * 0.99)) < 1)
    ok("最小方差", abs(min_variance(cov).sum() - 1) < 1e-6)
    ok("最大分散", abs(max_diversification(cov).sum() - 1) < 1e-6)
    ok("HRP", abs(hrp(cov).sum() - 1) < 1e-6)
    ok("BL", len(black_litterman(np.array([0.05] * 4), cov, np.eye(1, 4, 0),
                                np.array([0.08]))) == 4)


# ===== 迭代 51-60：风险/绩效/组合/信号扩展 =====
def test_iter_51_60():
    from lianghua.risk.drawdown import max_drawdown_info
    from lianghua.risk.beta import beta, alpha_annual
    from lianghua.risk.correlation import average_correlation
    from lianghua.perf.sortino import sortino, calmar, omega_ratio
    from lianghua.perf.dist import skew, kurtosis, value_at_risk
    from lianghua.perf.annualize import annual_return
    from lianghua.portfolio.target_vol import target_vol_weights
    from lianghua.portfolio.cppi import cppi
    from lianghua.signal.score import composite_score
    from lianghua.factor.zscore import winsorize, neutralize
    info = max_drawdown_info(CLOSE)
    ok("回撤信息", info["max_drawdown"] <= 0 and "peak_date" in info)
    ok("Beta", -3 < beta(RET, RET * 0.9) < 3)
    ok("Alpha", isinstance(alpha_annual(RET, RET), float))
    ok("平均相关", -1 <= average_correlation(RSETS) <= 1)
    ok("Sortino", isinstance(sortino(CLOSE), float))
    ok("Calmar", isinstance(calmar(CLOSE), float))
    ok("Omega", omega_ratio(CLOSE) > 0)
    ok("偏度峰度", isinstance(skew(CLOSE), float) and isinstance(kurtosis(CLOSE), float))
    ok("VaR", isinstance(value_at_risk(CLOSE), float))
    ok("年化收益", isinstance(annual_return(CLOSE), float))
    w_tv = target_vol_weights(RSETS, 0.12)
    port_v = float(np.sqrt(w_tv @ RSETS.cov().values @ w_tv) * np.sqrt(252))
    ok("目标波动", abs(port_v - 0.12) < 1e-6)
    ok("CPPI", cppi(RET).shape[1] == 3)
    s1 = pd.Series(np.random.RandomState(7).normal(0, 1, 100))
    s2 = pd.Series(np.random.RandomState(8).normal(0, 1, 100))
    ok("复合分数", len(composite_score({"a": s1, "b": s2})) == 100)
    ind = pd.Series(np.random.RandomState(2).choice(["a", "b"], 100))
    ok("中性化", abs(neutralize(pd.Series(np.random.RandomState(9).normal(0, 1, 100)), ind).mean()) < 1e-6)
    ok("缩尾", winsorize(pd.Series(np.random.RandomState(9).normal(0, 1, 100))).between(-5, 5).all())


# ===== 迭代 61-72：策略扩展 =====
def test_iter_61_72():
    from lianghua.strategy.dual_momentum import dual_momentum
    from lianghua.strategy.pair import pair_signal
    from lianghua.strategy.kalman import kalman_hedge
    from lianghua.strategy.donchian import donchian_signal
    from lianghua.strategy.keltner import keltner_signal
    from lianghua.strategy.coppock import coppock_signal
    from lianghua.strategy.seasonal import month_signal
    from lianghua.strategy.vol_breakout import vol_breakout_signal
    from lianghua.indicators import rsi
    from lianghua.strategy.rsi_divergence import rsi_divergence
    from lianghua.strategy.parallel_sar import parabolic_sar, sar_signal
    from lianghua.strategy.adaptive import adaptive_vol_switch
    from lianghua.strategy.ou_mean_reversion import ou_signal
    dfr2 = pd.DataFrame(np.random.RandomState(9).normal(0.0005, 0.01, (N, 3)),
                        columns=["A", "B", "CASH"])
    ok("双重动量", dual_momentum(dfr2, cash_col="CASH").notna().all())
    ok("配对", set(pair_signal(CLOSE, X).unique()).issubset([-1, 0, 1]))
    ok("卡尔曼", len(kalman_hedge(CLOSE, X)) == N)
    ok("唐奇安", set(donchian_signal(CLOSE).unique()).issubset([-1, 0, 1]))
    ok("Keltner", set(keltner_signal(CLOSE, HIGH, LOW).unique()).issubset([-1, 0, 1]))
    ok("Coppock", set(coppock_signal(CLOSE).unique()).issubset([0, 1]))
    ok("季节", 0 <= month_signal(CLOSE).mean() <= 1)
    ok("波动突破", set(vol_breakout_signal(CLOSE).unique()).issubset([-1, 0, 1]))
    ok("RSI背离", (rsi_divergence(CLOSE, rsi(CLOSE)) != 0).any())
    ok("SAR", len(parabolic_sar(HIGH, LOW)) == N)
    ok("SAR信号", set(sar_signal(HIGH, LOW).unique()).issubset([-1, 1]))
    ok("自适应", set(adaptive_vol_switch(RET).unique()).issubset([0, 1]))
    ok("OU", set(ou_signal(CLOSE).unique()).issubset([-1, 0, 1]))


# ===== 迭代 73-84：执行/风控/数据/绩效 =====
def test_iter_73_84():
    from lianghua.execution.sim_order import simulate_orders
    from lianghua.execution.slippage import fill_price
    from lianghua.risk.position_sizing import kelly_fraction, fixed_fractional
    from lianghua.risk.stop_loss import trailing_stop
    from lianghua.risk.take_profit import take_profit_levels
    from lianghua.risk.limit_order import backtest_limit
    from lianghua.data.resample import resample_ohlcv
    from lianghua.data.feature_store import FeatureStore
    from lianghua.signal.scheduler import rebalance_dates
    from lianghua.backtest.multi_strategy import multi_strategy_backtest
    from lianghua.portfolio.risk_parity_ewm import risk_parity_ewm
    from lianghua.perf.bootstrap import bootstrap_metric
    from lianghua.perf.annualize import annual_return
    bars = pd.DataFrame({"date": DATE, "open": CLOSE * 0.999, "high": HIGH,
                        "low": LOW, "close": CLOSE})
    ok("市价单", simulate_orders(
        [{"side": "BUY", "price": 100, "type": "MARKET", "date": bars["date"].iloc[10]}],
        bars)[0]["filled"] is not None)
    ok("滑点成交价", fill_price(100, -1e6, 1e7, 0.1) < 100)
    ok("Kelly", 0 <= kelly_fraction(0.55, 2.0) <= 1)
    ok("固定分数", fixed_fractional(1e6, 0.02, 0.05) > 0)
    ok("跟踪止损", len(trailing_stop(CLOSE, 100, 0.1, 0.1)) == N)
    ok("止盈档", 0.5 in take_profit_levels(100, [0.1, 0.2], [0.5, 0.5]).values())
    lo = backtest_limit([{"side": "BUY", "price": 50, "date": bars["date"].iloc[5]}], bars)
    ok("限价单不成交", bool(lo["filled"].iloc[0]) is False)
    ok("周线聚合", len(resample_ohlcv(bars.set_index("date"), "W")) > 0)
    import tempfile
    fs = FeatureStore(db=os.path.join(tempfile.mkdtemp(), "f.db"))
    fs.add("AAA", pd.DataFrame({"date": ["2023-01-02"], "rsi": [55.0]}), ["rsi"])
    ok("特征读写", fs.get("AAA", "rsi").iloc[0] == 55.0)
    ok("再平衡日", len(rebalance_dates("2023-01-01", "2023-03-31", "M")) >= 1)
    sig = pd.Series(np.where(CLOSE.rolling(20).mean().shift(1) > CLOSE, 1, -1), index=CLOSE.index)
    r = multi_strategy_backtest(pd.DataFrame({"date": DATE, "close": CLOSE}), {"a": sig})
    ok("多策略回测", r["equity"].iloc[-1] > 0)
    ok("EWM风险平价", abs(risk_parity_ewm(RSETS).sum() - 1) < 1e-6)
    b = bootstrap_metric(CLOSE, annual_return)
    ok("Bootstrap", b["n"] == 500 and b["low_95"] <= b["high_95"])


# ===== 迭代 85-100：组合/信号/事件/报告/UI/ML/因子/版本 =====
def test_iter_85_100():
    from lianghua.portfolio.clustering import cluster_assets
    from lianghua.signal.meta import majority_vote, weighted_meta
    from lianghua.backtest.event_study import event_study
    from lianghua.report.export_all import export_backtest
    from lianghua.core.constants import VERSION, COLOR_UP
    from lianghua.strategy.ml_signal import ml_signal
    from lianghua.data.universe import get_universe
    from lianghua.risk.regime import detect_regime
    from lianghua.perf.rolling_corr import rolling_corr_equity
    from lianghua.report.tearsheet import tearsheet
    from lianghua.execution.risk_check import pre_trade_check
    from lianghua.factor.returns import factor_long_short, factor_ic
    from lianghua.signal.rank import rank_signal, rank_to_position
    from lianghua.factor.zscore import winsorize
    from lianghua.perf.contrib import time_contrib
    from lianghua.backtest.vectorized import vectorized_backtest
    from lianghua import __version__, CAPABILITIES
    ok("聚类", len(cluster_assets(RSETS, 2)) == 2)
    s1 = pd.Series(np.random.RandomState(7).choice([-1, 0, 1], N))
    s2 = pd.Series(np.random.RandomState(8).choice([-1, 0, 1], N))
    ok("多数表决", set(majority_vote({"a": s1, "b": s2}).unique()).issubset([-1, 0, 1]))
    ok("加权元", len(weighted_meta({"a": s1, "b": s2})) == N)
    es = event_study(RET, [DATE[100], DATE[300]], 10)
    ok("事件研究", len(es) == 2)
    res = vectorized_backtest(pd.DataFrame({"date": DATE, "close": CLOSE}),
                            pd.Series(1, index=CLOSE.index))
    import tempfile
    d = tempfile.mkdtemp()
    export_backtest(res, d, "bt")
    ok("报告导出", os.path.exists(os.path.join(d, "bt_equity.csv")))
    ok("版本常量", "live" in VERSION and COLOR_UP == "#ff4d4f")
    ok("ML信号", ml_signal(pd.DataFrame({"close": CLOSE})).iloc[-1] in (-1, 0, 1))
    ok("指数成分", len(get_universe("沪深300")) >= 1)
    ok("波动率区制", set(detect_regime(RET).unique()).issubset([0, 1]))
    ok("滚动相关", len(rolling_corr_equity(CLOSE, CLOSE * 1.05)) == N - 1)
    ok("Tearsheet", len(tearsheet(CLOSE)) >= 6)
    ok("交易前检查", pre_trade_check(
        {"symbol": "X", "side": "BUY", "qty": 1, "price": 10}, {},
        {"gross": 1e6, "max_position_pct": 0.1, "max_gross_pct": 0.5})["ok"])
    fct = pd.Series(np.random.RandomState(6).normal(0, 1, N))
    fwd = CLOSE.pct_change(5).shift(-5)
    ok("因子多空", isinstance(factor_long_short(fct, fwd).iloc[0], float))
    ok("因子IC", isinstance(factor_ic(fct, fwd), float))
    ok("排名信号", len(rank_signal({"a": s1, "b": s2})) == N)
    ok("排名仓位", set(rank_to_position(rank_signal({"a": s1, "b": s2})).unique()).issubset([-1, 0, 1]))
    ok("收益贡献", len(time_contrib(pd.Series([0.25] * 4, RSETS.columns), RSETS)) == N)
    ok("包版本", __version__ == VERSION and len(CAPABILITIES) >= 13)


if __name__ == "__main__":
    print("=== 迭代 31-100 集成测试 ===")
    test_iter_31_38()
    print("迭代31-38 ✓")
    test_iter_39_50()
    print("迭代39-50 ✓")
    test_iter_51_60()
    print("迭代51-60 ✓")
    test_iter_61_72()
    print("迭代61-72 ✓")
    test_iter_73_84()
    print("迭代73-84 ✓")
    test_iter_85_100()
    print("迭代85-100 ✓")
    print("\nALL_ITER_31_100_OK  ✅ 共 70 项断言通过")
