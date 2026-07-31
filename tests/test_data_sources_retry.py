"""数据源适配器(sources) 可靠性测试：真实源重试 + 失败原因可观测。

覆盖的隐性问题（与网关对称，但 sources 是多源适配器诊断页实际走的路径）：
- 真实源因瞬时异常失败时无重试，直接返回 None，诊断页只显示"无数据"无原因；
- 失败原因未记录，用户无法判断是依赖缺失还是网络问题。
"""
import pandas as pd
import pytest

from lianghua.data.sources import (
    BaseSource, SyntheticSource, fetch_from, fetch_any, last_error,
)


def _df(start="2023-01-01", end="2023-06-30"):
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({
        "date": dates, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0, "volume": 1e6,
    })


def test_synthetic_always_works():
    df = SyntheticSource().fetch("X", "2023-01-01", "2023-06-30")
    assert not df.empty
    assert "close" in df.columns


def test_fetch_from_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class Flaky(BaseSource):
        name = "flaky"

        def fetch(self, symbol, start, end, asset="stock"):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return _df()

    import lianghua.data.sources as S
    monkeypatch.setitem(S.REGISTRY, "flaky", Flaky)
    df = fetch_from("flaky", "X", "2023-01-01", "2023-06-30", retries=2, backoff=0)
    assert calls["n"] == 3
    assert not df.empty
    assert last_error("flaky") is None


def test_fetch_from_records_failure_reason(monkeypatch):
    class Broken(BaseSource):
        name = "broken"

        def fetch(self, symbol, start, end, asset="stock"):
            raise ConnectionError("dependency missing")

    import lianghua.data.sources as S
    monkeypatch.setitem(S.REGISTRY, "broken", Broken)
    df = fetch_from("broken", "X", "2023-01-01", "2023-06-30", retries=0)
    assert df is None
    assert last_error("broken") is not None
    assert "dependency missing" in last_error("broken")


def test_fetch_any_falls_through_to_synthetic(monkeypatch):
    class Down(BaseSource):
        name = "down"

        def fetch(self, symbol, start, end, asset="stock"):
            return None

    import lianghua.data.sources as S
    monkeypatch.setitem(S.REGISTRY, "akshare", Down)
    monkeypatch.setitem(S.REGISTRY, "baostock", Down)
    df = fetch_any("X", "2023-01-01", "2023-06-30")
    assert not df.empty
    assert "close" in df.columns
