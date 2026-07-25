"""Broker 统一接口与共享数据结构。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...core.assets import AssetType


@dataclass
class Order:
    """一笔订单。"""
    symbol: str
    side: str            # BUY / SELL
    qty: int
    price: float
    asset_type: AssetType = AssetType.STOCK

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "side": self.side,
                "qty": self.qty, "price": self.price,
                "asset_type": self.asset_type.value}


@dataclass
class Position:
    symbol: str
    asset_type: AssetType
    qty: int = 0          # 正=多, 负=空
    avg_price: float = 0.0
    margin_frozen: float = 0.0

    def update(self, side: str, qty: int, price: float, multiplier: float, margin_rate: float):
        """按订单更新持仓，采用正确的净头寸语义。

        反向下单先平仓（部分平仓保留原均价），仅当卖出量超过当前持仓时才
        翻转为反向新仓；避免「卖出等量多头却整手翻成空头」的隐性错误。
        """
        signed = qty if side == "BUY" else -qty
        if self.qty == 0:
            self.avg_price = price
            self.qty = signed
        elif (self.qty > 0) == (signed > 0):
            # 同方向加仓：更新加权均价
            self.avg_price = (self.avg_price * self.qty + price * signed) / (self.qty + signed)
            self.qty += signed
        else:
            # 反方向：先平仓，剩余部分才开反向新仓
            if abs(signed) < abs(self.qty):
                self.qty += signed  # 部分平仓，方向/均价不变
            elif abs(signed) == abs(self.qty):
                self.qty = 0
                self.avg_price = 0.0
            else:
                remainder = signed + self.qty  # 与 signed 同号
                self.avg_price = price
                self.qty = remainder
        if self.asset_type == AssetType.FUTURE:
            self.margin_frozen = abs(self.qty) * price * multiplier * margin_rate
        elif self.asset_type == AssetType.OPTION:
            self.margin_frozen = abs(self.qty) * price * multiplier  # 权利金占用


class BaseBroker(ABC):
    """所有 broker 的统一抽象接口。"""

    def __init__(self, init_cash: float = 1_000_000.0):
        self.init_cash = init_cash
        self.cash = init_cash
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self._oid = 0

    def _next_id(self) -> int:
        self._oid += 1
        return self._oid

    def connect(self) -> bool:
        """连接柜台（模拟 broker 默认可用）。"""
        return True

    @abstractmethod
    def submit(self, order: Order) -> dict:
        ...

    def cancel(self, order_id) -> bool:
        return False

    @abstractmethod
    def get_account(self, market_values: dict[str, float] | None = None) -> dict:
        ...

    def exercise_option(self, symbol: str, underlying_price: float) -> dict:
        raise NotImplementedError(f"{type(self).__name__} 不支持期权行权")

    def redeem_fund(self, symbol: str, nav: float) -> dict:
        raise NotImplementedError(f"{type(self).__name__} 不支持基金赎回")
