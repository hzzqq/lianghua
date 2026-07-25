"""迭代 181–200（超自驱动目标深化）集成测试：
期权组合 / 数据多源 / 执行增强 / 指标 tech4 / 绩效 extra2 / 风险 extra / 因子 combine 扩展。

全部离线可跑：数据用 synthetic 源或内置随机行情，无外网依赖。
"""
import sys
import os
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from lianghua.option.strategy import (OPTION_COMBO_REGISTRY, get_option_combo, payoff_curve)
from lianghua.data.sources import list_sources, fetch_from, fetch_any
from lianghua.data.universe import list_universes, get_universe
from lianghua.execution.brokers import make_broker, REGISTRY as BROKER_REG
from lianghua.execution.orders import bracket_order, evaluate_bracket
from lianghua.execution.pre_trade import pre_trade_check
from lianghua.indicators import INDICATOR_FUNCS
from lianghua.indicators.tech4 import (adx, pivot, heikin_ashi, renko, demarker, klinger,
                                       chaikin_volatility, force_index, know_sure_thing,
                                       zigzag, price_channels)
from lianghua.perf.extra2 import (payoff_ratio, hit_ratio, profit_factor,
                                  outlier_ratio, recovery_factor)
from lianghua.risk.extra import (entropy_risk, concentration_risk, correlation_risk,
                                 expected_shortfall, liquidity_adjusted_var,
                                 regime_var, stress_var, max_loss_prob)
from lianghua.factor.combine import (factor_corr, factor_orthogonalize,
                                     factor_weight_decay, factor_cross_section,
                                     factor_portfolio, factor_decay_halflife)
from lianghua.core.capabilities import summary_counts, list_capabilities
from lianghua.core.constants import VERSION
from lianghua.core.assets import AssetType


def _demo_df(n: int = 120):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    return pd.DataFrame({"date": dates, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1e6})


def test_version():
    assert VERSION == "1.6.0-live", VERSION


def test_option_combos():
    names = ["covered_call", "protective_put", "collar", "straddle", "strangle",
             "iron_butterfly", "calendar_spread", "ratio_spread", "diagonal_spread",
             "long_call", "long_put", "vertical_spread", "iron_condor", "butterfly"]
    for nm in names:
        assert nm in OPTION_COMBO_REGISTRY, nm
    legs = get_option_combo("covered_call")
    assert isinstance(legs, list) and len(legs) == 2
    pc = payoff_curve(legs, 80, 120)
    assert len(pc) == 200 and "pnl" in pc
    # 股票腿在组合损益中线性生效（S=120 的损益 > S=80）
    grid = payoff_curve(legs, 80, 120)
    assert grid.iloc[-1]["pnl"] > grid.iloc[0]["pnl"]


def test_data_sources():
    assert {"synthetic", "akshare", "baostock"}.issubset(set(list_sources()))
    df = fetch_from("synthetic", "600519.SH", "2023-01-01", "2023-06-30")
    assert df is not None and not df.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    df2 = fetch_any("600519.SH", "2023-01-01", "2023-06-30")
    assert df2 is not None and not df2.empty
    assert "沪深300" in list_universes()
    assert len(get_universe("沪深300")) == 10


def test_execution():
    assert "paper" in BROKER_REG
    b = make_broker("paper", slippage=0.001)
    o = bracket_order("600519.SH", "BUY", 100, 100.0, 95.0, 108.0, asset_type=AssetType.STOCK)
    res = b.submit(o["entry"])
    assert res["slippage"] > 0, "PaperBroker 应施加滑点"
    fired = evaluate_bracket(o, 101.0)
    assert "entry" in fired
    acct = {"cash": 1_000_000, "equity": 1_000_000, "positions": {}}
    ok, _ = pre_trade_check(o["entry"], acct, {"max_order_value": 1e7})
    assert ok
    ok2, _ = pre_trade_check(o["entry"], acct, {"max_order_value": 100})
    assert not ok2


def test_indicators_tech4():
    df = _demo_df()
    for fn in (adx, pivot, heikin_ashi, renko, demarker, klinger, chaikin_volatility,
               force_index, know_sure_thing, zigzag, price_channels):
        out = fn(df)
        assert len(out) == len(df), fn.__name__
    for nm in ["adx", "pivot", "heikin_ashi", "renko", "demarker", "klinger",
               "chaikin_volatility", "force_index", "know_sure_thing",
               "zigzag", "price_channels"]:
        assert nm in INDICATOR_FUNCS


def test_perf_extra2():
    eq = _demo_df()["close"].astype(float)
    for fn in (payoff_ratio, hit_ratio, profit_factor, outlier_ratio, recovery_factor):
        v = fn(eq)
        assert np.isfinite(v), fn.__name__


def test_risk_extra():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0005, 0.015, 200))
    w = np.array([0.5, 0.3, 0.2])
    assert np.isfinite(entropy_risk(w))
    assert 0.0 <= concentration_risk(w) <= 1.0
    assert np.isfinite(correlation_risk(pd.DataFrame(rng.normal(0, 1, (200, 3)))))
    assert expected_shortfall(rets) <= 0
    assert liquidity_adjusted_var(rets, 1_000_000) >= 0
    mask = rng.normal(0, 1, 200) < 0
    assert np.isfinite(regime_var(rets, mask))
    assert np.isfinite(stress_var(rets, 0.2))
    assert 0.0 <= max_loss_prob(rets) <= 1.0


def test_factor_combine_extra():
    rng = np.random.default_rng(2)
    panel = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=[f"A{i}" for i in range(5)])
    fwd = panel.shift(-1)
    assert np.isfinite(factor_corr(panel, fwd))
    orth = factor_orthogonalize(panel, panel)
    assert orth.shape == panel.shape
    assert len(factor_weight_decay(panel)) == 5
    z = factor_cross_section(panel)
    assert z.shape == panel.shape
    w = factor_portfolio(panel, top=0.2)
    assert w.abs().sum() > 0
    ic = factor_decay_halflife(pd.Series([0.1, 0.05, 0.025, 0.0125]))
    assert np.isfinite(ic) or ic == float("inf")


def test_capabilities_counts():
    sc = summary_counts()
    cap = list_capabilities()
    assert len(cap["indicators"]) == 37
    assert len(cap["option_combos"]) == 14
    assert sc["perf_functions"] >= 57
    assert sc["risk_functions"] >= 51
    assert sc["factor_functions"] >= 26
    print("✓ 能力计数：", sc)


if __name__ == "__main__":
    test_version(); test_option_combos(); test_data_sources(); test_execution()
    test_indicators_tech4(); test_perf_extra2(); test_risk_extra()
    test_factor_combine_extra(); test_capabilities_counts()
    print("\n迭代181-200 测试全部通过 ✅")
