"""lianghua 多资产框架集成测试。

运行：python -m pytest tests/ 或 python tests/test_all_assets.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lianghua.core.assets import AssetType, detect_asset_type, get_contract_spec
from lianghua.data.gateway import DataGateway
from lianghua.backtest.runner import run_backtest, SUPPORTED
from lianghua.perf.metrics import report, attribution, sharpe, max_drawdown
from lianghua.risk.manager import RiskManager
from lianghua.execution.broker import SimBroker, Order
from lianghua.option.pricing import bs_price, greeks


def _finite(s: pd.Series) -> bool:
    return bool(np.isfinite(s).all())


def test_asset_detection():
    assert detect_asset_type("600519.SH") == AssetType.STOCK
    assert detect_asset_type("110011.OF") == AssetType.FUND
    assert detect_asset_type("RB2410.SHF") == AssetType.FUTURE
    assert detect_asset_type("510050C3000.SH") == AssetType.OPTION


def test_gateway_all_assets():
    gw = DataGateway()
    specs = {
        "600519.SH": AssetType.STOCK,
        "110011.OF": AssetType.FUND,
        "RB2410.SHF": AssetType.FUTURE,
        "510050C3000.SH": AssetType.OPTION,
    }
    for sym, at in specs.items():
        df = gw.fetch(sym, "2024-01-01", "2024-03-31", asset=at)
        assert len(df) > 0
        if at == AssetType.OPTION:
            assert "option_price" in df.columns and "delta" in df.columns
        else:
            assert {"open", "high", "low", "close"}.issubset(df.columns)


def test_option_pricing():
    p = bs_price(100, 100, 0.25, 0.02, 0.2, "CALL")
    assert p > 0
    g = greeks(100, 100, 0.25, 0.02, 0.2, "CALL")
    assert 0 < g["delta"] < 1
    assert g["gamma"] > 0


def test_backtest_all_assets():
    cases = [
        ("600519.SH", "stock", "sma_cross"),
        ("110011.OF", "fund", "sma_cross"),
        ("110011.OF", "fund", "dca"),
        ("RB2410.SHF", "future", "breakout"),
        ("510050C3000.SH", "option", "option_call_trend"),
    ]
    for sym, asset, strat in cases:
        res = run_backtest(sym, "2023-01-01", "2024-12-31", strategy=strat, asset=asset)
        assert len(res.equity) > 0
        assert _finite(res.equity), f"{sym}/{strat} 净值含非有限值"
        perf = report(res.equity, res.trades)
        assert math.isfinite(perf["sharpe"])
        assert math.isfinite(perf["max_drawdown"])


def test_risk_differentiated():
    rm = RiskManager()
    assert rm.max_leverage(AssetType.FUTURE) > 1
    assert rm.max_premium_frac(AssetType.OPTION) <= 1
    # 期货保证金超强平线应被拦截
    assert rm.check_margin(used_margin=900000, equity=1000000, asset_type=AssetType.FUTURE) is False


def test_broker_multi_asset():
    b = SimBroker(1_000_000)
    b.submit(Order("RB2410.SHF", "BUY", 10, 3000.0, AssetType.FUTURE))
    acct = b.get_account()
    assert acct["margin_frozen"] > 0
    b.submit(Order("510050C3000.SH", "BUY", 5, 0.2, AssetType.OPTION))
    ex = b.exercise_option("510050C3000.SH", underlying_price=3.2)
    assert ex["status"] == "exercised"


def test_attribution():
    eqs = {}
    for sym, asset, strat in [("600519.SH", "stock", "sma_cross"),
                              ("RB2410.SHF", "future", "breakout")]:
        r = run_backtest(sym, "2023-01-01", "2024-12-31", strategy=strat, asset=asset)
        eqs[sym] = r.equity
    df = attribution(eqs)
    assert len(df) == 2
    assert "contribution" in df.columns


def test_gateway_option_cache():
    """真实期权盘口（含希腊字母）必须落本地缓存，且二次 fetch 命中缓存拿到同一份。

    此前本用例直接用默认的 data_cache.db 打真实网络，有两个隐患：
    - 断言"期权已落缓存"其实是靠库里遗留的 65 行 source=demo 老数据蒙混过关的。
      合成/演示期权早已明确不落缓存（见 fetch 里 src != "demo" 的守卫），
      换一台干净机器 clone 下来 option_bars 是空的，这条断言必挂。
    - 两次 fetch 都打真实网络：上游抖动时一次走 BS 合成(58 行)、一次走纯演示(65 行)，
      len(df1) == len(df2) 随机失败。
    改为用隔离的临时缓存 + 桩真实源，只验真正要保证的契约，不再依赖网络与残留状态。
    """
    import os
    import sqlite3
    import tempfile

    sym = "510050C3000.SH"
    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    real = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "underlying": 2.5,
        "option_price": 0.1, "delta": 0.5, "gamma": 0.1, "vega": 0.2,
        "theta": -0.01, "rho": 0.05, "strike": 3.0,
        "expiry": "2024-06-28", "type": "CALL",
    })
    fd, cache_db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        gw = DataGateway(cache_db=cache_db)
        gw._from_akshare = lambda symbol, start, end, asset=None: real.copy()

        df1 = gw.fetch(sym, "2024-01-01", "2024-03-31", asset=AssetType.OPTION)
        assert len(df1) > 0 and not gw.last_was_demo
        assert {"delta", "gamma", "vega", "theta", "rho"} <= set(df1.columns)

        con = sqlite3.connect(cache_db)
        n = con.execute("SELECT COUNT(*) FROM option_bars WHERE symbol=?",
                        (sym,)).fetchone()[0]
        con.close()
        assert n > 0, "真实期权希腊字母应落本地缓存"

        # 二次 fetch 必须命中缓存：真实源换成会抛异常的桩，只有命中缓存才可能成功
        def _must_not_be_called(*a, **k):
            raise AssertionError("已有完整缓存时不应再打真实源")

        gw._from_akshare = _must_not_be_called
        df2 = gw.fetch(sym, "2024-01-01", "2024-03-31", asset=AssetType.OPTION)
        assert gw.last_source == "cache"
        assert len(df1) == len(df2)
    finally:
        os.remove(cache_db)


def test_broker_factory_and_live_adapters():
    from lianghua.execution import make_broker
    b = make_broker("sim", init_cash=1_000_000)
    b.submit(Order("600519.SH", "BUY", 100, 100.0, AssetType.STOCK))
    assert b.get_account()["cash"] == 1_000_000 - 100 * 100.0
    # 实盘适配器未连接下单应抛清晰异常，证明可插拔切换
    q = make_broker("qmt", account_id="123")
    try:
        q.submit(Order("600519.SH", "BUY", 100, 100.0, AssetType.STOCK))
        assert False, "QMT 未连接却下单成功"
    except (NotImplementedError, RuntimeError):
        pass


def test_factor_search():
    from lianghua.factor import FactorSearcher, FactorEngine
    gw = DataGateway()
    df = gw.fetch("600519.SH", "2023-01-01", "2024-12-31", asset=AssetType.STOCK)
    top = FactorSearcher(df).search(top_n=5)
    assert len(top) > 0
    assert "ic" in top[0] and "factor" in top[0]
    for r in top:
        assert math.isfinite(r["ic"]) or pd.isna(r["ic"])
    eq = FactorEngine(df).backtest_factor(FactorEngine(df).compute(top[0]["factor"]))
    assert len(eq) == len(df)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    raise SystemExit(1 if failed else 0)
