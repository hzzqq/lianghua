"""核心类型与资产元数据。"""
from .assets import (
    AssetType,
    ContractSpec,
    detect_asset_type,
    get_contract_spec,
    EXCHANGE_FUTURE,
)

__all__ = [
    "AssetType",
    "ContractSpec",
    "detect_asset_type",
    "get_contract_spec",
    "EXCHANGE_FUTURE",
]
