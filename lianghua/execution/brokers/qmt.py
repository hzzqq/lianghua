"""迅投 QMT (MiniQMT/xtquant) 实盘适配器（第十二轮·连接实盘）。

依赖：本机安装 MiniQMT 客户端 + xtquant SDK（券商版）。
未安装 SDK 或未配置账户时 connect() 抛出明确 RuntimeError（优雅降级，
不影响 sim/paper 路径）。

用法::

    b = QmtBroker(path=r"D:/qmt/userdata_mini", account_id="8888888888")
    b.connect()
    b.submit(Order("600519.SH", "BUY", 100, 1700.0))
"""
from __future__ import annotations

from ...core.assets import AssetType
from .base import BaseBroker, Order


class QmtBroker(BaseBroker):
    """真实柜台：迅投 QMT。所有下单直接进交易所，务必先在 paper 验证策略。"""

    def __init__(self, path: str = "", account_id: str = "",
                 account_type: str = "STOCK", session_id: int = 20260723,
                 init_cash: float = 0.0):
        super().__init__(init_cash=init_cash)
        self.path = path
        self.account_id = account_id
        self.account_type = account_type
        self.session_id = int(session_id)
        self._trader = None
        self._account = None
        self._xtconstant = None
        self.connected = False

    # ---------- 连接 ----------
    def connect(self) -> bool:
        if self.connected:
            return True
        if not self.account_id:
            raise RuntimeError("QmtBroker 需要 account_id（资金账号）")
        try:
            from xtquant import xtconstant  # noqa: F401
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
        except ImportError as e:
            raise RuntimeError(
                "未安装 xtquant SDK（需 MiniQMT 客户端 + 券商版 xtquant），"
                "无法连接 QMT 真实柜台：%s" % e) from e
        self._xtconstant = xtconstant
        trader = XtQuantTrader(self.path, self.session_id)
        trader.start()
        if trader.connect() != 0:
            raise RuntimeError(
                "QMT 连接失败：path=%r，请确认 MiniQMT 已登录" % self.path)
        account = StockAccount(self.account_id, self.account_type)
        trader.subscribe(account)
        self._trader, self._account = trader, account
        self.connected = True
        return True

    def _require(self):
        if not self.connected:
            self.connect()

    def _order_type(self, order: Order) -> int:
        c = self._xtconstant
        at, side = order.asset_type, order.side
        if at == AssetType.FUTURE:
            return (c.FUTURE_OPEN_BUY if side == "BUY" else c.FUTURE_OPEN_SELL)
        if at == AssetType.OPTION:
            return (getattr(c, "OPTION_BUY", c.STOCK_BUY) if side == "BUY"
                    else getattr(c, "OPTION_SELL", c.STOCK_SELL))
        return c.STOCK_BUY if side == "BUY" else c.STOCK_SELL

    # ---------- 交易 ----------
    def submit(self, order: Order) -> dict:
        self._require()
        c = self._xtconstant
        oid = self._trader.order_stock(
            self._account, order.symbol, self._order_type(order),
            int(order.qty), c.FIX_PRICE, float(order.price),
            "lianghua", "")
        status = "submitted" if oid >= 0 else "rejected"
        return {"order_id": int(oid), "status": status,
                "symbol": order.symbol, "side": order.side,
                "qty": order.qty, "price": order.price,
                "fill_price": order.price, "broker": "qmt"}

    def cancel(self, order_id) -> bool:
        self._require()
        return self._trader.cancel_order_stock(self._account, int(order_id)) == 0

    # ---------- 查询 ----------
    def get_account(self, market_values: dict[str, float] | None = None) -> dict:
        self._require()
        asset = self._trader.query_stock_asset(self._account)
        poss = self._trader.query_stock_positions(self._account) or []
        positions = {p.stock_code: int(p.volume) for p in poss}
        pos_value = float(sum(getattr(p, "market_value", 0.0) for p in poss))
        cash = float(getattr(asset, "cash", 0.0)) if asset else 0.0
        return {"cash": round(cash, 2),
                "position_value": round(pos_value, 2),
                "margin_frozen": 0.0,
                "equity": round(cash + pos_value, 2),
                "positions": positions}

    def query_positions(self) -> list[dict]:
        self._require()
        poss = self._trader.query_stock_positions(self._account) or []
        return [{"symbol": p.stock_code, "qty": int(p.volume),
                 "avg_price": float(getattr(p, "open_price", 0.0)),
                 "market_value": float(getattr(p, "market_value", 0.0))}
                for p in poss]

    def query_orders(self) -> list[dict]:
        self._require()
        orders = self._trader.query_stock_orders(self._account) or []
        return [{"order_id": int(o.order_id), "symbol": o.stock_code,
                 "qty": int(o.order_volume), "price": float(o.price),
                 "status": int(o.order_status)} for o in orders]

    def __repr__(self):
        return "QmtBroker(account=%r, connected=%s)" % (
            self.account_id, self.connected)
