"""纸面券商：在 SimBroker 基础上叠加滑点模拟（迭代 181+，超自驱动目标）。

用于无真实柜台的策略预演：提交即按滑点调整成交价，其余账户逻辑与
SimBroker 完全一致（现金/持仓/保证金），保证可无缝替换。
"""
from __future__ import annotations

from .sim import SimBroker


class PaperBroker(SimBroker):
    """纸面撮合：成交价 = 委托价 ± 滑点（买高卖低），其余同 SimBroker。"""

    def __init__(self, init_cash: float = 1_000_000.0, slippage: float = 0.0005):
        super().__init__(init_cash=init_cash)
        self.slippage = float(slippage)

    def submit(self, order):
        slip = order.price * self.slippage
        eff_price = order.price + slip if order.side == "BUY" else order.price - slip
        eff_price = max(eff_price, 1e-6)
        oid = self._next_id()
        from .base import Order
        eff = Order(order.symbol, order.side, order.qty, round(eff_price, 4),
                    asset_type=order.asset_type)
        res = super().submit(eff)
        res["fill_price"] = round(eff_price, 4)
        res["order_id"] = oid
        res["slippage"] = round(float(slip), 6)
        return res

    def __repr__(self):
        return f"PaperBroker(cash={self.cash:.0f}, slippage={self.slippage})"
