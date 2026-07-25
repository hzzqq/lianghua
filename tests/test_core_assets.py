"""core/assets 模块打磨测试：资产识别、合约规格校验、保证金计算、能力发现。"""
from __future__ import annotations

import pytest

from lianghua.core.assets import (
    AssetType,
    ContractSpec,
    describe_symbol,
    detect_asset_type,
    get_contract_spec,
    normalize_symbol,
)


def test_detect_asset_type():
    assert detect_asset_type("600000.SH") == AssetType.STOCK
    assert detect_asset_type("510050.OF") == AssetType.FUND
    assert detect_asset_type("RB2410.CFE") == AssetType.FUTURE
    assert detect_asset_type("510050C003000.SH") == AssetType.OPTION
    assert detect_asset_type("  au990.sh  ") == AssetType.STOCK  # 空白忽略


def test_get_contract_spec_future_known():
    cs = get_contract_spec("RB2410.CFE")
    assert cs.asset_type == AssetType.FUTURE
    assert cs.multiplier == 10
    assert 0 < cs.margin_rate <= 1.0


def test_get_contract_spec_future_unknown_defaults():
    cs = get_contract_spec("ZZ9999.DCE")
    assert cs.asset_type == AssetType.FUTURE
    # 未知品种回落到安全默认（multiplier=10, margin_rate=0.10）
    assert cs.multiplier == 10
    assert cs.margin_rate == 0.10


def test_get_contract_spec_option_etf_strike_scaled():
    cs = get_contract_spec("510050C003000.SH")
    assert cs.asset_type == AssetType.OPTION
    assert cs.opt_type == "CALL"
    assert cs.strike == pytest.approx(3.0)  # 003000 / 1000
    assert cs.point_value == 10000.0


def test_get_contract_spec_stock_full_cash():
    cs = get_contract_spec("600000.SH")
    assert cs.asset_type == AssetType.STOCK
    assert cs.multiplier == 1.0
    assert cs.margin_rate == 1.0


def test_validate_rejects_bad_margin():
    with pytest.raises(ValueError):
        ContractSpec(
            asset_type=AssetType.FUTURE, symbol="X",
            multiplier=10, margin_rate=0.0,
        ).validate()
    with pytest.raises(ValueError):
        ContractSpec(
            asset_type=AssetType.FUTURE, symbol="X",
            multiplier=0, margin_rate=0.1,
        ).validate()


def test_validate_rejects_bad_option():
    with pytest.raises(ValueError):
        ContractSpec(
            asset_type=AssetType.OPTION, symbol="X",
            multiplier=10000, margin_rate=1.0,
            strike=0.0, opt_type="CALL",
        ).validate()
    with pytest.raises(ValueError):
        ContractSpec(
            asset_type=AssetType.OPTION, symbol="X",
            multiplier=10000, margin_rate=1.0,
            strike=3.0, opt_type="WEIRD",
        ).validate()


def test_required_margin():
    cs = get_contract_spec("RB2410.CFE")
    # 名义金额 100000 * 0.10 = 10000
    assert cs.required_margin(100_000.0) == pytest.approx(10_000.0)
    assert cs.required_margin(200_000.0) == pytest.approx(20_000.0)


def test_describe_symbol():
    d = describe_symbol("510050C003000.SH")
    assert d["asset_type"] == "option"
    assert d["opt_type"] == "CALL"
    assert d["strike"] == pytest.approx(3.0)
    assert d["point_value"] == 10000.0
    assert d["symbol"] == "510050C003000.SH"


def test_normalize_symbol():
    assert normalize_symbol("  rb2410.cfe ") == "RB2410.CFE"


def test_get_contract_spec_never_returns_invalid():
    # 所有已知路径产出的规格都应通过校验
    for sym in ["600000.SH", "510050.OF", "RB2410.CFE", "IF2406.CFE",
                "510050C003000.SH", "AU2412.SHF"]:
        cs = get_contract_spec(sym)
        cs.validate()  # 不应抛错
