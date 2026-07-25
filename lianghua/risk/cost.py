"""精细化交易成本模型。

统一管理：比例佣金、固定费用、滑点、市场冲击、印花税(卖出)、最低费用。
可按资产类别选择预设（build_cost），也可自定义 CostModel。
引擎在撮合时优先使用 CostModel；未传入时回退到引擎自有的简化成本。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    commission_rate: float = 0.0003   # 比例佣金（成交额占比，万三）
    fixed_fee: float = 0.0            # 每笔固定费用
    slippage: float = 0.001           # 滑点（比例）
    impact: float = 0.0               # 市场冲击（成交额占比）
    stamp_tax: float = 0.0            # 印花税（仅卖出，成交额占比）
    min_fee: float = 0.0              # 单笔最低费用

    def fill_price(self, price: float, side: str) -> float:
        """考虑滑点后的成交价。买/多/平空加滑点，卖/空/平多减滑点。"""
        buyish = side in ("BUY", "LONG", "COVER")
        adj = self.slippage if buyish else -self.slippage
        return price * (1.0 + adj)

    def cost_for(self, side: str, price: float, size: float, multiplier: float = 1.0) -> float:
        """该笔交易的总成本（金额，正=支出，含佣金/滑点/冲击/印花税）。"""
        notional = abs(price * size * multiplier)
        slip = notional * self.slippage
        comm = notional * self.commission_rate + self.fixed_fee
        imp = notional * self.impact
        fee = slip + comm + imp
        tax = notional * self.stamp_tax if side in ("SELL", "SHORT", "COVER") else 0.0
        return max(fee + tax, self.min_fee)


# 预设（贴近 A 股/期货/期权/基金常见费率）
STOCK_COST = CostModel(commission_rate=0.0003, slippage=0.001, stamp_tax=0.001, min_fee=5.0)
FUTURE_COST = CostModel(commission_rate=0.000025, slippage=0.0005, impact=0.00001)
OPTION_COST = CostModel(commission_rate=0.0003, slippage=0.001)
FUND_COST = CostModel(commission_rate=0.0015, slippage=0.0)   # 申购费代理(一次性)


_PRESETS = {
    "stock": STOCK_COST, "equity": STOCK_COST,
    "future": FUTURE_COST, "futures": FUTURE_COST,
    "option": OPTION_COST, "options": OPTION_COST,
    "fund": FUND_COST, "funds": FUND_COST,
}


def build_cost(preset: str | None = None, **overrides) -> "CostModel | None":
    """按资产类别预设构造成本模型；None 表示默认(引擎自有简化成本)。"""
    if preset is None:
        return None
    base = _PRESETS.get(str(preset).lower())
    if base is None:
        raise ValueError(f"未知成本预设 {preset}，可选 {list(_PRESETS)}")
    if overrides:
        return CostModel(**{**base.__dict__, **overrides})
    return base
