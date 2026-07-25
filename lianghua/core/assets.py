"""资产类型定义与合约元数据。

支持四类可量化资产：股票(stock) / 基金(fund) / 期货(future) / 期权(option)。
统一通过 symbol 路由到对应数据生成器与回测引擎。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class AssetType(str, Enum):
    STOCK = "stock"
    FUND = "fund"
    FUTURE = "future"
    OPTION = "option"


# 期货交易所后缀
EXCHANGE_FUTURE = {"SHF", "CFE", "DCE", "CZC", "INE", "GFE"}

# 标的默认合约规格（演示用，可扩展）
_DEFAULT_SPECS = {
    # 商品期货：每手乘数（吨/克/桶）、保证金率
    "RB": dict(multiplier=10, margin_rate=0.10, commission_per_hand=3.0),   # 螺纹钢 10吨/手
    "AU": dict(multiplier=1000, margin_rate=0.08, commission_per_hand=10.0), # 黄金 1000克/手
    "CU": dict(multiplier=5, margin_rate=0.10, commission_per_hand=0.0),    # 铜 5吨/手
    "IF": dict(multiplier=300, margin_rate=0.12, commission_per_hand=0.0),  # 股指 300元/点
    "IC": dict(multiplier=200, margin_rate=0.12, commission_per_hand=0.0),
    # ETF 期权：每手 10000 份，权利金按点=元
    "510050": dict(multiplier=10000, margin_rate=1.0, commission_per_hand=1.6),
    "510300": dict(multiplier=10000, margin_rate=1.0, commission_per_hand=1.6),
}


@dataclass
class ContractSpec:
    """合约规格，驱动保证金/乘数/手续费计算。"""

    asset_type: AssetType
    symbol: str
    underlying: str = ""
    multiplier: float = 1.0        # 每点价值（期货：元/点；期权：元/点）
    contract_size: float = 1.0     # 每手合约乘数（股/份/吨/克/桶）
    margin_rate: float = 1.0       # 保证金率（1=全额，期货<1）
    commission_per_hand: float = 0.0
    strike: float = 0.0            # 期权行权价（元/点）
    opt_type: str = ""             # CALL / PUT
    expiry: str = ""               # 到期日 YYYY-MM-DD
    strike_scale: float = 1.0      # 行权价编码缩放（ETF期权=1000）

    @property
    def point_value(self) -> float:
        """每跳动一点对应的合约价值。"""
        return self.multiplier * self.contract_size

    def validate(self) -> None:
        """校验合约规格合法性，非法时抛 ValueError。

        隐性修复：原先规格由默认 dict 拼接，若配置错误（如 margin_rate=0）
        会被静默接受，导致保证金计算产出 0、等于无限杠杆而无人察觉。
        """
        if self.multiplier <= 0:
            raise ValueError(f"multiplier 必须为正，得到 {self.multiplier}")
        if self.contract_size <= 0:
            raise ValueError(f"contract_size 必须为正，得到 {self.contract_size}")
        if not (0 < self.margin_rate <= 1.0):
            raise ValueError(f"margin_rate 必须在 (0,1]，得到 {self.margin_rate}")
        if self.commission_per_hand < 0:
            raise ValueError(
                f"commission_per_hand 不能为负，得到 {self.commission_per_hand}"
            )
        if self.asset_type == AssetType.OPTION:
            if self.strike <= 0:
                raise ValueError(f"期权 strike 必须为正，得到 {self.strike}")
            if self.opt_type not in ("CALL", "PUT"):
                raise ValueError(f"期权 opt_type 必须 CALL/PUT，得到 {self.opt_type}")

    def required_margin(self, notional: float) -> float:
        """按当前规格计算某名义金额所需的保证金（新增能力）。"""
        return float(notional) * self.margin_rate


def normalize_symbol(symbol: str) -> str:
    """归一化 symbol：去空白、转大写。"""
    return symbol.strip().upper()


def describe_symbol(symbol: str, asset_type: AssetType | None = None) -> dict:
    """返回 symbol 的资产类型与关键合约规格，供 UI/CLI 能力发现（新增能力）。"""
    at = asset_type or detect_asset_type(symbol)
    spec = get_contract_spec(symbol, at)
    return {
        "symbol": normalize_symbol(symbol),
        "asset_type": at.value,
        "underlying": spec.underlying,
        "multiplier": spec.multiplier,
        "contract_size": spec.contract_size,
        "margin_rate": spec.margin_rate,
        "point_value": spec.point_value,
        "opt_type": spec.opt_type or None,
        "strike": spec.strike or None,
    }


def detect_asset_type(symbol: str) -> AssetType:
    """根据 symbol 自动识别资产类别。"""
    s = symbol.strip().upper()
    if s.endswith(".OF"):
        return AssetType.FUND
    parts = s.rsplit(".", 1)
    exch = parts[-1] if len(parts) > 1 else ""
    if exch in EXCHANGE_FUTURE:
        return AssetType.FUTURE
    # 期权：基础码 + C/P + 行权价数字 + 后缀
    if re.match(r"^[A-Z0-9]+[CP]\d+\.[A-Z]+$", s):
        return AssetType.OPTION
    return AssetType.STOCK


def _parse_option(symbol: str) -> dict:
    """解析期权 symbol，返回底层/类型/行权价。

    若 symbol 不是标准期权合约格式（如用户直接传入标的代码 510050.SH
    并指定 asset=option），则退化为一个默认平值期权（CALL，ETF 默认
    行权价 3.0 / 非 ETF 默认 100.0），便于离线演示与真实标的锚定。
    """
    s = symbol.strip().upper()
    m = re.match(r"^(?P<base>[A-Z0-9]+?)(?P<otype>[CP])(?P<strike>\d+)\.(?P<exch>[A-Z]+)$", s)
    if not m:
        base = s.split(".")[0]
        is_etf = base.isdigit() and len(base) == 6
        return {"underlying": base, "opt_type": "CALL",
                "strike": 3.0 if is_etf else 100.0,
                "strike_scale": 1000.0 if is_etf else 1.0}
    base = m.group("base")
    otype = "CALL" if m.group("otype") == "C" else "PUT"
    strike_raw = int(m.group("strike"))
    # ETF 期权（6位纯数字标的）行权价编码/1000
    if base.isdigit() and len(base) == 6:
        strike = strike_raw / 1000.0
        scale = 1000.0
    else:
        strike = float(strike_raw)
        scale = 1.0
    return {"underlying": base, "opt_type": otype, "strike": strike, "strike_scale": scale}


def get_contract_spec(symbol: str, asset_type: AssetType | None = None) -> ContractSpec:
    """构造某 symbol 的合约规格。"""
    at = asset_type or detect_asset_type(symbol)
    if at == AssetType.OPTION:
        info = _parse_option(symbol)
        base = info["underlying"]
        spec = _DEFAULT_SPECS.get(base, dict(multiplier=10000, margin_rate=1.0))
        cs = ContractSpec(
            asset_type=at, symbol=symbol, underlying=base,
            multiplier=spec.get("multiplier", 10000),
            margin_rate=spec.get("margin_rate", 1.0),
            commission_per_hand=spec.get("commission_per_hand", 0.0),
            strike=info["strike"], opt_type=info["opt_type"],
            strike_scale=info["strike_scale"],
        )
        cs.validate()
        return cs
    if at == AssetType.FUTURE:
        # 取字母前缀作为品种，如 RB2410 -> RB
        m = re.match(r"^([A-Z]+)", symbol.upper())
        product = m.group(1) if m else symbol.upper()
        spec = _DEFAULT_SPECS.get(product, dict(multiplier=10, margin_rate=0.10))
        cs = ContractSpec(
            asset_type=at, symbol=symbol, underlying=symbol,
            multiplier=spec.get("multiplier", 10),
            margin_rate=spec.get("margin_rate", 0.10),
            commission_per_hand=spec.get("commission_per_hand", 0.0),
        )
        cs.validate()
        return cs
    # 股票 / 基金：全额，1 手=1 单位
    cs = ContractSpec(
        asset_type=at, symbol=symbol, underlying=symbol,
        multiplier=1.0, contract_size=1.0, margin_rate=1.0,
    )
    cs.validate()
    return cs
