"""Round 6：RiskManager 差异化止损 + None 安全 + 访问器/快照。"""
import pytest

from lianghua.core.assets import AssetType
from lianghua.risk.manager import RiskManager, DEFAULT_LIMITS


def test_check_exit_uses_asset_specific_stop_loss():
    rm = RiskManager()
    # 股票默认止损 10%：跌 12% 应触发
    assert rm.check_exit(100, 88, AssetType.STOCK) is True
    # 期权默认止损 50%：跌 12% 不应触发
    assert rm.check_exit(100, 88, AssetType.OPTION) is False
    # 期权跌 55% 触发
    assert rm.check_exit(100, 45, AssetType.OPTION) is True


def test_check_exit_none_uses_stock_default():
    rm = RiskManager(stop_loss=0.20)
    # 构造参数对 STOCK 生效；跌 15% 在 20% 内不触发
    assert rm.check_exit(100, 85) is False
    assert rm.check_exit(100, 78) is True


def test_check_exit_rejects_bad_entry():
    rm = RiskManager()
    assert rm.check_exit(0, 90) is False
    assert rm.check_exit(-5, 90) is False


def test_limits_for_none_is_safe():
    rm = RiskManager()
    # 此前 None 会 KeyError
    assert rm.limits_for(None)["stop_loss"] == 0.10
    assert rm.max_position_for(None) == 0.95


def test_max_position_for_differentiated():
    rm = RiskManager()
    assert rm.max_position_for(AssetType.FUTURE) == 0.90
    assert rm.max_position_for(AssetType.OPTION) == 0.50
    assert rm.max_position_for(AssetType.STOCK) == 0.95


def test_drawdown_for_accessor():
    rm = RiskManager()
    assert rm.drawdown_for(AssetType.FUTURE) == 0.30
    assert rm.drawdown_for(None) == 0.25


def test_position_size_returns_bounded_fraction():
    rm = RiskManager()
    assert rm.position_size(1000, 10) == pytest.approx(0.95)
    assert rm.position_size(0, 10) == 0.0
    assert rm.position_size(1000, 0) == 0.0
    rm2 = RiskManager(limits={AssetType.OPTION: {"max_position": 0.3}})
    assert rm2.position_size(1000, 10, AssetType.OPTION) == pytest.approx(0.3)


def test_check_margin_edge_equity_zero():
    rm = RiskManager()
    # 权益为 0 时，仅允许零保证金占用
    assert rm.check_margin(0, 0, AssetType.FUTURE) is True
    assert rm.check_margin(1, 0, AssetType.FUTURE) is False


def test_premium_ok_rejects_negative():
    rm = RiskManager()
    assert rm.premium_ok(-1, 100, AssetType.OPTION) is False
    assert rm.premium_ok(10, 100, AssetType.OPTION) is True  # <= 20%


def test_summary_snapshot():
    rm = RiskManager(blacklist=["AAA"])
    s = rm.summary()
    assert s["blacklist"] == ["AAA"]
    assert "stock" in s["limits"]


def test_instance_limits_isolated():
    rm1 = RiskManager(limits={AssetType.FUTURE: {"max_leverage": 3.0}})
    rm2 = RiskManager()
    # 各实例独立，互不影响
    assert rm1.max_leverage(AssetType.FUTURE) == 3.0
    assert rm2.max_leverage(AssetType.FUTURE) == 5.0
