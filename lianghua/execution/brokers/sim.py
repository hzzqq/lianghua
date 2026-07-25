"""本地模拟券商：记录订单与持仓，提供账户快照。支持四类资产。"""
from __future__ import annotations

from ...core.assets import AssetType, get_contract_spec
from .base import BaseBroker, Order, Position


class SimBroker(BaseBroker):
    """模拟撮合网关，离线可用，接口与实盘 broker 一致。"""

    def connect(self) -> bool:
        return True

    def validate_order(self, order: Order) -> tuple[bool, str]:
        """校验订单合法性，返回 (ok, reason)。

        - qty 必须为正整数；
        - price 必须为正；
        - side 仅允许 BUY / SELL。
        非法订单不应进入撮合，避免产生错误方向与负现金。
        """
        if not isinstance(order.qty, int) or order.qty <= 0:
            return False, f"qty 必须为正整数，收到 {order.qty!r}"
        if not isinstance(order.price, (int, float)) or order.price <= 0:
            return False, f"price 必须为正，收到 {order.price!r}"
        if order.side not in ("BUY", "SELL"):
            return False, f"side 非法: {order.side!r}"
        return True, ""

    def _cash_delta(self, order: Order, spec, mult: float) -> float:
        """估算该订单对现金的净变动（用于额度预检）。

        期货：开多/开空均冻结保证金 -> 现金净流出；
        期权：买入付权利金(流出) / 卖出收权利金(流入)；
        股票/基金：买入付全款(流出) / 卖出收款项(流入)。
        """
        if order.asset_type == AssetType.FUTURE:
            return -order.qty * order.price * mult * spec.margin_rate
        if order.asset_type == AssetType.OPTION:
            premium = order.qty * order.price * mult
            return -premium if order.side == "BUY" else premium
        # 股票 / 基金：全额
        return -order.qty * order.price if order.side == "BUY" else order.qty * order.price

    def submit(self, order: Order) -> dict:
        ok, reason = self.validate_order(order)
        if not ok:
            return {"status": "rejected", "reason": reason, "order_id": None}
        # 现金/保证金额度预检：不允许产生负现金（隐性超额下单陷阱）
        spec = get_contract_spec(order.symbol, order.asset_type)
        mult = spec.multiplier
        delta = self._cash_delta(order, spec, mult)
        if self.cash + delta < -1e-9:
            return {"status": "rejected",
                    "reason": "现金/保证金不足",
                    "order_id": None,
                    "cash_after": round(self.cash, 2)}

        oid = self._next_id()
        self.orders.append(order)
        pos = self.positions.setdefault(
            order.symbol, Position(order.symbol, order.asset_type))

        if order.asset_type == AssetType.FUTURE:
            # 期货开多/开空均冻结保证金（现金净流出），纠正原 SELL 误把保证金存入现金的 bug
            self.cash -= order.qty * order.price * mult * spec.margin_rate
            pos.update(order.side, order.qty, order.price, mult, spec.margin_rate)
        elif order.asset_type == AssetType.OPTION:
            premium = order.qty * order.price * mult
            if order.side == "BUY":
                self.cash -= premium
            else:
                self.cash += premium
            pos.update(order.side, order.qty, order.price, mult, 1.0)
        else:  # 股票/基金：全额
            if order.side == "BUY":
                self.cash -= order.qty * order.price
            else:
                self.cash += order.qty * order.price
            pos.update(order.side, order.qty, order.price, 1.0, 1.0)

        # 数值稳定：账本现金保留 2 位，避免浮点漂移累积
        self.cash = round(self.cash, 2)
        return {"status": "filled", "fill_price": order.price,
                "order_id": oid, "cash_after": self.cash}

    def max_affordable(self, symbol: str, price: float,
                       asset_type: AssetType | str = "stock") -> int:
        """测算当前现金在给定价格下最多可买数量（不含手续费）。

        期货按保证金占用、期权按权利金占用折算；返回非负整数，资金不足时返回 0。
        用于下单前的头寸规模控制，是撮合引擎的可观测/可控新能力。
        """
        at = AssetType(asset_type) if isinstance(asset_type, str) else asset_type
        if price <= 0:
            return 0
        spec = get_contract_spec(symbol, at)
        mult = spec.multiplier
        if at == AssetType.FUTURE:
            per = price * mult * spec.margin_rate
        elif at == AssetType.OPTION:
            per = price * mult
        else:
            per = price
        if per <= 0:
            return 0
        return max(0, int(self.cash // per))

    def exercise_option(self, symbol: str, underlying_price: float) -> dict:
        """期权行权（到期时用标的价结算内在价值）。"""
        pos = self.positions.get(symbol)
        if not pos or pos.qty == 0:
            return {"status": "no_position"}
        spec = get_contract_spec(symbol, AssetType.OPTION)
        intrinsic = max(underlying_price - spec.strike, 0.0) if spec.opt_type == "CALL" \
            else max(spec.strike - underlying_price, 0.0)
        settle = pos.qty * intrinsic * spec.multiplier
        self.cash += settle
        self.positions.pop(symbol)
        return {"status": "exercised", "settle": round(settle, 2), "cash_after": round(self.cash, 2)}

    def redeem_fund(self, symbol: str, nav: float) -> dict:
        """基金赎回：按净值结清份额。"""
        pos = self.positions.get(symbol)
        if not pos or pos.qty == 0:
            return {"status": "no_position"}
        proceed = pos.qty * nav
        self.cash += proceed
        self.positions.pop(symbol)
        return {"status": "redeemed", "proceed": round(proceed, 2), "cash_after": round(self.cash, 2)}

    def get_account(self, market_values: dict[str, float] | None = None) -> dict:
        """账户快照：现金 + 持仓市值 + 保证金占用。"""
        mv = market_values or {}
        pos_value = sum(mv.get(s, 0.0) for s in self.positions)
        margin = sum(p.margin_frozen for p in self.positions.values())
        equity = self.cash + pos_value
        return_rate = (equity - self.init_cash) / self.init_cash if self.init_cash else 0.0
        return {
            "cash": round(self.cash, 2),
            "position_value": round(pos_value, 2),
            "margin_frozen": round(margin, 2),
            "equity": round(equity, 2),
            "return_rate": round(return_rate, 6),
            "positions": {s: p.qty for s, p in self.positions.items()},
        }
