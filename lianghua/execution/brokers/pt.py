"""恒生 PTrade 实盘适配器（第十二轮·连接实盘）。

PTrade 策略通常运行在券商托管环境内（ptrade API 由环境注入/提供）。
本适配器把 BaseBroker 接口映射到 ptrade 的 order/get_positions 等原语；
本地无 ptrade 模块时 connect() 抛出明确 RuntimeError（优雅降级）。
"""
from __future__ import annotations

from .base import BaseBroker, Order


class PtBroker(BaseBroker):
    """真实柜台：恒生 PTrade。amount 正=买入 负=卖出。"""

    def __init__(self, account_id: str = "", init_cash: float = 0.0, api=None):
        super().__init__(init_cash=init_cash)
        self.account_id = account_id
        self._api = api           # 允许注入（托管环境 / 测试桩）
        self.connected = api is not None

    # ---------- 连接 ----------
    def connect(self) -> bool:
        if self.connected:
            return True
        try:
            import ptrade  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "未检测到 ptrade 运行环境（PTrade 策略需运行在券商托管环境，"
                "或注入 api 桩）：%s" % e) from e
        self._api = ptrade
        self.connected = True
        return True

    def _require(self):
        if not self.connected:
            self.connect()

    # ---------- 交易 ----------
    def submit(self, order: Order) -> dict:
        self._require()
        amount = int(order.qty) if order.side == "BUY" else -int(order.qty)
        style = None
        limit_order = getattr(self._api, "LimitOrder", None)
        if limit_order is not None:
            style = limit_order(float(order.price))
        if style is not None:
            oid = self._api.order(order.symbol, amount, style=style)
        else:
            oid = self._api.order(order.symbol, amount)
        status = "submitted" if oid is not None else "rejected"
        return {"order_id": oid, "status": status,
                "symbol": order.symbol, "side": order.side,
                "qty": order.qty, "price": order.price,
                "fill_price": order.price, "broker": "pt"}

    def cancel(self, order_id) -> bool:
        self._require()
        fn = getattr(self._api, "cancel_order", None)
        if fn is None:
            return False
        try:
            fn(order_id)
            return True
        except Exception:
            return False

    # ---------- 查询 ----------
    def get_account(self, market_values: dict[str, float] | None = None) -> dict:
        self._require()
        cash = 0.0
        get_asset = getattr(self._api, "get_asset", None)
        if get_asset is not None:
            asset = get_asset()
            if isinstance(asset, dict):
                cash = float(asset.get("cash", 0.0))
            else:
                cash = float(getattr(asset, "cash", 0.0))
        positions, pos_value = {}, 0.0
        get_positions = getattr(self._api, "get_positions", None)
        if get_positions is not None:
            for p in (get_positions() or []):
                sym = getattr(p, "sid", None) or getattr(p, "symbol", "")
                qty = int(getattr(p, "amount", 0) or getattr(p, "qty", 0))
                mv = float(getattr(p, "market_value", 0.0))
                positions[sym] = qty
                pos_value += mv
        return {"cash": round(cash, 2),
                "position_value": round(pos_value, 2),
                "margin_frozen": 0.0,
                "equity": round(cash + pos_value, 2),
                "positions": positions}

    def query_positions(self) -> list[dict]:
        self._require()
        out = []
        get_positions = getattr(self._api, "get_positions", None)
        if get_positions is not None:
            for p in (get_positions() or []):
                out.append({"symbol": getattr(p, "sid", "") or getattr(p, "symbol", ""),
                            "qty": int(getattr(p, "amount", 0) or getattr(p, "qty", 0)),
                            "market_value": float(getattr(p, "market_value", 0.0))})
        return out

    def query_orders(self) -> list[dict]:
        self._require()
        fn = getattr(self._api, "get_orders", None)
        if fn is None:
            return []
        return [dict(o) if isinstance(o, dict) else {"order": str(o)}
                for o in (fn() or [])]

    def __repr__(self):
        return "PtBroker(account=%r, connected=%s)" % (
            self.account_id, self.connected)
