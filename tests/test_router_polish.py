"""R42: router 导入与内省验收。"""
import numpy as np

from lianghua.execution.router import AccountRouter, MultiLiveEngine
from lianghua.core.assets import AssetType


def test_all_no_import_error():
    # 原 __all__ 含不存在的 'rebalance' 会导致 ImportError；现应干净
    import importlib
    import lianghua.execution.router as m
    importlib.reload(m)
    for name in m.__all__:
        assert hasattr(m, name), f"__all__ 引用了不存在的名字 {name}"


def test_route_by_asset_then_prefix_then_default():
    # 优先级：by_asset > by_prefix > default
    rt = AccountRouter({"by_asset": {"future": "B"},
                        "by_prefix": {"600": "C"}}, default="A")
    assert rt.route("IF2401", AssetType.FUTURE) == "B"      # by_asset 命中
    assert rt.route("600519.XSHG", AssetType.STOCK) == "C"  # by_prefix 命中
    assert rt.route("000999.XSHG", AssetType.STOCK) == "A"  # 走默认


def test_route_preview_batch():
    rt = AccountRouter({"by_prefix": {"600": "C"}}, default="A")
    prev = rt.preview(["600519.XSHG", "000001.XSHE"])
    assert prev == {"600519.XSHG": "C", "000001.XSHE": "A"}


def test_invalid_route_value_raises():
    try:
        AccountRouter({"by_asset": {"stock": None}})
    except ValueError:
        return
    raise AssertionError("空/非字符串路由值应抛 ValueError")


def test_rebalance_status_before_call_is_none():
    eng = MultiLiveEngine({}, AccountRouter(default="A"), symbols=[])
    assert eng.rebalance_status() is None


class _StubBroker:
    __name__ = "PaperBroker"

    def get_account(self, mv):
        return {"equity": 1_000_000.0}


class _StubEngine:
    def __init__(self):
        self.broker = _StubBroker()
        self.capital = 1_000_000.0


def test_rebalance_target_weights_validation():
    eng = MultiLiveEngine({"A": _StubEngine()}, AccountRouter(default="A"))
    try:
        eng.rebalance(target_weights={"A": np.nan})
    except ValueError:
        return
    raise AssertionError("非有限 target_weights 应抛 ValueError")


def test_rebalance_status_records_last():
    eng = MultiLiveEngine({"A": _StubEngine()}, AccountRouter(default="A"))
    out = eng.rebalance(target_weights={"A": 1.0})
    assert eng.rebalance_status() is not None
    assert out["plan"][0]["account"] == "A"
    assert out["plan"][0]["target_equity"] == 1_000_000.0
