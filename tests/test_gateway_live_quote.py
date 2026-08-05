"""live_quote 实时快照测试：演示(假)数据降级必须如实标注 source=demo，不得冒充 last_close。

隐性正确性 bug：live_quote 在真实行情不可用时降级到 fetch 的收盘价，却把来源标成
"last_close"；实盘引擎若据此做市值标记，会用假价格冒充真实收盘价，用户无感知。

**本模块必须与网络隔离**：live_quote 会先打 AKShare 实时接口，只有它失败才走到
被测的降级分支。早先的版本没有屏蔽这一步，于是"能不能过"取决于跑测试的机器有没有
外网——断网时全绿、联网时两条降级用例直接红（实测 source 变成 akshare）。
这种用例比没有用例更糟：它让套件在 CI 与开发机之间随机变色，最终被当噪音忽略。
"""
import sys
import types

import pandas as pd
import pytest

from lianghua.data.gateway import DataGateway


@pytest.fixture
def gw():
    return DataGateway(cache_db=":memory:")


@pytest.fixture(autouse=True)
def offline_spot(monkeypatch):
    """把 AKShare 实时快照接口钉成"不可用"，强制走降级分支且不发任何网络请求。

    live_quote 内部是 `import akshare as ak` 的惰性导入，所以这里直接改
    sys.modules 里的 akshare（没装则塞一个假模块），monkeypatch 会在用例结束后还原。
    """
    def _down(*_a, **_k):
        raise RuntimeError("offline: 实时行情接口在测试中被禁用")

    ak = sys.modules.get("akshare")
    if ak is None:
        ak = types.ModuleType("akshare")
        monkeypatch.setitem(sys.modules, "akshare", ak)
    monkeypatch.setattr(ak, "stock_zh_a_spot_em", _down, raising=False)
    monkeypatch.setattr(ak, "fund_etf_spot_em", _down, raising=False)


def _df_with_source(source: str):
    return pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "open": [100.0, 101.0], "high": [101.0, 102.0],
        "low": [99.0, 100.0], "close": [100.5, 101.5], "volume": [1e6, 1e6],
        "source": [source, source],
    })


def test_live_quote_labels_demo_not_last_close(gw):
    """fetch 降级为演示(假)数据时，live_quote 的 source 必须标 demo。"""
    gw.fetch = lambda *a, **k: _df_with_source("demo")
    q = gw.live_quote("600519.SH")
    assert q["source"] == "demo", "演示(假)数据降级必须如实标 demo，不得冒充真实收盘"
    assert gw.last_was_demo is True


def test_live_quote_labels_real_as_last_close(gw):
    """fetch 返回真实日线时，live_quote 的 source 应标 last_close（非 demo）。"""
    gw.fetch = lambda *a, **k: _df_with_source("akshare")
    q = gw.live_quote("600519.SH")
    assert q["source"] == "last_close"
    assert gw.last_was_demo is False


def test_live_quote_total_failure_does_not_leave_stale_demo_flag(gw):
    """实时源与日线双双失败时，必须把 last_source 刷成 none，而不是留着上次的值。

    真实危害：先对 A 做了一次降级到演示数据的 fetch（last_was_demo=True），再对 B 调
    live_quote 且彻底失败——B 拿到 price=NaN，但调用方读 last_was_demo 得到的是 A 的
    True/上一次的 False，两种都错：要么误报"这是演示数据"，要么让"根本没取到价"
    这件事完全无声。UI 的 gw_fetch 与实盘市值标记都只看这两个字段。
    """
    def _boom(*_a, **_k):
        raise RuntimeError("日线也挂了")

    gw.last_source, gw.last_was_demo = "demo", True  # 上一次无关调用留下的陈旧状态
    gw.fetch = _boom
    q = gw.live_quote("600519.SH")
    assert q["source"] == "none"
    assert pd.isna(q["price"])
    assert gw.last_source == "none", "彻底失败必须刷新 last_source，不能沿用上次的值"
    assert gw.last_was_demo is False
    assert any("live_quote" in m for m in gw.last_warnings()), "失败原因必须可观测，不能被吞掉"


def test_live_quote_real_time_hit_refreshes_stale_demo_flag(gw, monkeypatch):
    """实时源命中时，必须把上一次遗留的 last_was_demo=True 清掉。

    否则 UI 会对着一条货真价实的实时报价挂"⚠️ 已降级为演示(假)数据"的横幅，
    用户从此不再相信这个提示——降级告警一旦有假阳性就等于失效。
    """
    spot = pd.DataFrame({"代码": ["600519"], "最新价": [1688.0]})
    monkeypatch.setattr(sys.modules["akshare"], "stock_zh_a_spot_em",
                        lambda *a, **k: spot, raising=False)

    gw.last_source, gw.last_was_demo = "demo", True
    q = gw.live_quote("600519.SH")
    assert q == {"symbol": "600519.SH", "price": 1688.0, "source": "akshare"}
    assert gw.last_source == "akshare"
    assert gw.last_was_demo is False, "真实实时报价被误标成演示数据，降级告警会变成假阳性"
