"""live_quote 实时快照测试：演示(假)数据降级必须如实标注 source=demo，不得冒充 last_close。

隐性正确性 bug：live_quote 在真实行情不可用时降级到 fetch 的收盘价，却把来源标成
"last_close"；实盘引擎若据此做市值标记，会用假价格冒充真实收盘价，用户无感知。
"""
import pandas as pd
import pytest

from lianghua.data.gateway import DataGateway


@pytest.fixture
def gw():
    return DataGateway(cache_db=":memory:")


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
