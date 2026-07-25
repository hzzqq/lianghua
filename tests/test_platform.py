"""平台化能力集成测试（对应"都做"三件事）：
- ③ 统一编排器 run_plan（多资产一键编排）
- ② 数据网关真实/演示来源可追溯（source 列）
- ① UI 全部页面函数无头渲染（桩 streamlit）

离线安全：编排器与网关测试用桩网关/强制演示路径，不依赖外网；
UI 测试用桩 streamlit 直接调用页面函数。
"""
import sys
import os
import datetime
import warnings

import numpy as np
import pandas as pd

# 确保项目根在路径中（直接 `python tests/xxx.py` 时脚本目录不会被识别为包根）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from lianghua.backtest.orchestrator import run_plan
from lianghua.data.gateway import DataGateway
from lianghua.core.assets import AssetType


# ---------------- ③ 统一编排器 ----------------
def test_orchestrator_multi_asset():
    class StubGW(DataGateway):
        def fetch(self, symbol, start, end, freq="daily", asset=None, timeout=15.0):
            a = asset or __import__("lianghua.core.assets",
                                    fromlist=["detect_asset_type"]).detect_asset_type(symbol)
            return self._demo_data(symbol, start, end, a)

    plan = [
        {"symbol": "600519.SH", "asset": "stock", "strategy": "sma_cross",
         "weight": 0.4, "start": "2023-01-01", "end": "2023-12-31"},
        {"symbol": "510300.SH", "asset": "fund", "strategy": "momentum",
         "weight": 0.4, "start": "2023-01-01", "end": "2023-12-31"},
        {"symbol": "RB0.SHF", "asset": "future", "strategy": "breakout",
         "weight": 0.2, "start": "2023-01-01", "end": "2023-12-31"},
    ]
    r = run_plan(plan, init_cash=1_000_000, gw=StubGW())
    assert isinstance(r["equity"], pd.Series) and len(r["equity"]) > 100
    assert set(r["weights"].keys()) == {"600519.SH", "510300.SH", "RB0.SHF"}
    for k in ("final_equity", "sharpe", "max_drawdown", "annual_return"):
        assert k in r["metrics"], f"组合指标缺 {k}"
    assert len(r["components"]) == 3
    assert r["equity"].iloc[0] == 1_000_000
    print("✓ 编排器多资产组合 OK")


# ---------------- ② 网关来源可追溯 ----------------
def test_gateway_source_column():
    class DemoGW(DataGateway):
        def _from_csv(self, symbol):
            return None

        def _from_akshare(self, *a, **k):
            return None

        def _from_baostock(self, *a, **k):
            return None

    gw = DemoGW()
    df = gw.fetch("TEST.SH", "2023-01-01", "2023-06-30", asset=AssetType.STOCK)
    assert not df.empty
    cached = gw._cache_get("TEST.SH", "2023-01-01", "2023-06-30", AssetType.STOCK)
    assert cached is not None and "source" in cached.columns
    assert str(cached["source"].dropna().iloc[0]) == "demo"
    print("✓ 网关 source 列追溯 OK（demo）")


# ---------------- ① UI 页面无头渲染 ----------------
def test_ui_all_pages_smoke():
    # 注入桩 streamlit，直接调用各页面函数验证胶水无 NameError
    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def metric(self, *a, **k):
            pass

        def __getattr__(self, n):
            return lambda *a, **k: None

    class _St:
        sidebar = None

        def __init__(self):
            self.sidebar = self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def set_page_config(self, *a, **k):
            pass

        def title(self, *a, **k):
            pass

        def header(self, *a, **k):
            pass

        def subheader(self, *a, **k):
            pass

        def caption(self, *a, **k):
            pass

        def divider(self, *a, **k):
            pass

        def info(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

        def success(self, *a, **k):
            pass

        def json(self, *a, **k):
            pass

        def metric(self, *a, **k):
            pass

        def dataframe(self, *a, **k):
            pass

        def plotly_chart(self, *a, **k):
            pass

        def image(self, *a, **k):
            pass

        def radio(self, *a, **k):
            opts = k.get("options") or (a[1] if len(a) > 1 else None)
            return opts[0] if opts else ""

        def selectbox(self, *a, **k):
            opts = k.get("options") or (a[1] if len(a) > 1 else None)
            return opts[0] if opts else ""

        def text_input(self, *a, **k):
            return ""

        def text_area(self, *a, **k):
            return ""

        def date_input(self, *a, **k):
            return datetime.date(2023, 1, 1)

        def number_input(self, *a, **k):
            return 0

        def button(self, *a, **k):
            return False

        def checkbox(self, *a, **k):
            return False

        def slider(self, *a, **k):
            # slider("x", min, max, default) -> 返回 default；无则 0
            return a[3] if len(a) > 3 else k.get("value", 0)

        def download_button(self, *a, **k):
            return None

        def line_chart(self, *a, **k):
            return None

        def spinner(self, *a, **k):
            return _Col()

        def columns(self, n=1, *a, **k):
            return tuple(_Col() for _ in range(int(n)))

        def tabs(self, n=1, *a, **k):
            if not isinstance(n, int):
                n = len(n) if hasattr(n, "__len__") else 1
            return tuple(_Col() for _ in range(int(n)))

        def write(self, *a, **k):
            return None

    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = _St()
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "lh_app_smoke", "lianghua/ui/app.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        # 全部页面函数（含 迭代181+ 新增 指标实验室/绩效风险分析/数据执行能力/能力总览）
        page_fns = [n for n in dir(app) if n.startswith("page_") and callable(getattr(app, n))]
        assert len(page_fns) >= 19, f"UI 页面数应≥19，实际 {len(page_fns)}"
        for fn in ["page_single", "page_basket", "page_orchestrator", "page_option",
                   "page_future_spread", "page_fund", "page_var", "page_vectorized",
                   "page_notify", "page_montecarlo", "page_param", "page_html",
                   "page_strategies", "page_portfolio", "page_factor",
                   "page_indicator_lab", "page_perf_risk", "page_data_exec", "page_capabilities"]:
            getattr(app, fn)()
    finally:
        if saved is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = saved
    print(f"✓ UI {len(page_fns)} 个页面渲染无异常")


if __name__ == "__main__":
    test_orchestrator_multi_asset()
    test_gateway_source_column()
    test_ui_all_pages_smoke()
    print("\n全部平台集成测试通过 ✅")
