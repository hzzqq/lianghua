"""UI 数据健康摘要（cache_health_summary）与两行图（make_2row）单元测试。

R19-A：首页数据健康按「行级真实 vs 演示残留」呈现（R18 list_cached 的 demo_rows），
含残留时给出 ⚠️ 提示与一键清洗入口（purge_demo_cache）。
R19-B：make_2row 空数据优雅降级（不再裸渲染空白坐标轴）、柱按逐柱涨跌方向红涨绿跌着色。

导入方式沿用 test_platform 的桩 streamlit 方案：app.py 模块尾部有 PAGES[PAGE]() 顶层执行，
必须用可配置的桩（radio/selectbox 返回真实页面名）；同时把后端地址指到不可达端口，
保证 page_home 的数据健康块走离线分支——测试绝不触真实后端/缓存库。
"""
import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _make_stub_streamlit():
    sm = MagicMock()
    sm.session_state = {}
    sm.radio.return_value = "首页"      # 模块尾 PAGES[PAGE]() 需要合法页面名
    sm.selectbox.return_value = "仪表盘"  # 侧栏模块选择
    # st.columns(2) 会被解包为 (c1, c2)：按入参返回对应数量的容器 mock
    def _columns(spec, *a, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(n)]
    sm.columns.side_effect = _columns
    return sm


@pytest.fixture(scope="module")
def app_mod():
    saved_st = sys.modules.get("streamlit")
    saved_env = os.environ.get("LIANGHUA_DATA_BACKEND")
    # 不可达端口（port 1）：后端探针立即失败，数据健康块走离线分支
    os.environ["LIANGHUA_DATA_BACKEND"] = "http://127.0.0.1:1"
    sys.modules["streamlit"] = _make_stub_streamlit()
    try:
        spec = importlib.util.spec_from_file_location(
            "lh_app_cache_health", os.path.join(ROOT, "lianghua", "ui", "app.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        if saved_st is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = saved_st
        if saved_env is None:
            os.environ.pop("LIANGHUA_DATA_BACKEND", None)
        else:
            os.environ["LIANGHUA_DATA_BACKEND"] = saved_env


# ---------------- cache_health_summary ----------------

def test_summary_mixed_rows(app_mod):
    cache = {
        "A.SH": {"rows": 10, "demo_rows": 3, "source": "akshare"},
        "B.SH": {"rows": 5, "demo_rows": 5, "source": "akshare"},
        "C.SH": {"rows": 7, "demo_rows": 0, "source": "tencent"},
    }
    h = app_mod.cache_health_summary(cache)
    assert h["symbols"] == 3
    assert h["real_rows"] == 14   # 7 + 0 + 7
    assert h["demo_rows"] == 8    # 3 + 5 + 0
    assert h["polluted"] == ["A.SH"]
    assert h["demo_symbols"] == ["B.SH"]


def test_summary_legacy_without_demo_rows(app_mod):
    """旧后端（无 demo_rows 字段）兼容：全部视为真实行。"""
    cache = {"A.SH": {"rows": 10, "source": "akshare"}}
    h = app_mod.cache_health_summary(cache)
    assert h["symbols"] == 1 and h["real_rows"] == 10 and h["demo_rows"] == 0
    assert h["polluted"] == [] and h["demo_symbols"] == []


def test_summary_empty_and_none(app_mod):
    for cache in (None, {}, {"X": "not-a-dict"}):
        h = app_mod.cache_health_summary(cache)
        assert h["symbols"] == 0 and h["real_rows"] == 0 and h["demo_rows"] == 0


def test_summary_clamps_bad_demo_rows(app_mod):
    """demo_rows > rows 的脏数据必须被钳制，不得出现负的真实行数。"""
    h = app_mod.cache_health_summary({"A.SH": {"rows": 2, "demo_rows": 9}})
    assert h["demo_rows"] == 2 and h["real_rows"] == 0
    assert h["demo_symbols"] == ["A.SH"]


# ---------------- make_2row ----------------

def _equity(n=50, drift=0.002):
    idx = pd.bdate_range("2024-01-01", periods=n)
    vals = 100.0 * np.cumprod(1.0 + np.full(n, drift))
    return pd.Series(vals, index=idx)


def test_make_2row_normal_two_traces(app_mod):
    fig = app_mod.make_2row(_equity(), _equity(drift=-0.001), "净值", "回撤")
    assert len(fig.data) == 2
    # 主序列按首尾方向着色：上涨红
    assert fig.data[0].line.color == "#ff4d4f"


def test_make_2row_bar_direction_colors(app_mod):
    """柱按逐柱相对前值方向着色：涨红/跌绿（A 股语义）。"""
    vals = pd.Series([1.0, 2.0, 1.5, 3.0, 2.0],
                     index=pd.bdate_range("2024-01-01", periods=5))
    fig = app_mod.make_2row(_equity(), vals, "净值", "持仓")
    colors = list(fig.data[1].marker.color)
    assert colors == ["#ff4d4f", "#ff4d4f", "#00d486", "#ff4d4f", "#00d486"]


def test_make_2row_empty_degrades_gracefully(app_mod):
    """空/全 NaN 主序列：返回空态占位图而非裸坐标轴，且不抛异常。"""
    empty = pd.Series(dtype=float)
    for s1 in (empty, pd.Series([np.nan] * 5), None):
        fig = app_mod.make_2row(s1, s1, "净值", "回撤")
        assert len(fig.data) == 0, "空态应是纯注解占位图"
        assert any("暂无可绘制数据" in a.text for a in fig.layout.annotations)


def test_make_2row_empty_second_series_single_trace(app_mod):
    """次序列为空：只画主序列，不渲染空 trace。"""
    fig = app_mod.make_2row(_equity(), pd.Series(dtype=float), "净值", "回撤")
    assert len(fig.data) == 1
