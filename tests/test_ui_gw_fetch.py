"""UI 取数告警测试：确保真实行情失败降级到演示(假)数据时，gw_fetch 返回可提示文案。

这是上一轮网关可观测能力的 UI 落地——避免用户拿假数据跑出的结论被误当真实行情。
"""
from lianghua.ui.app import gw_fetch


class _FakeGW:
    def __init__(self, demo: bool, returns):
        self.last_was_demo = demo
        self._returns = returns

    def fetch(self, *a, **k):
        return self._returns


def test_gw_fetch_warns_on_demo():
    gw = _FakeGW(demo=True, returns="DF")
    df, warn = gw_fetch(gw, "600519.SH", "2024-01-01", "2024-01-10")
    assert df == "DF"
    assert warn, "演示(假)数据必须返回非空告警文案"


def test_gw_fetch_no_warn_on_real():
    gw = _FakeGW(demo=False, returns="DF")
    df, warn = gw_fetch(gw, "600519.SH", "2024-01-01", "2024-01-10")
    assert df == "DF"
    assert warn == "", "真实数据不应产生告警"
