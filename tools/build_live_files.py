# -*- coding: utf-8 -*-
"""一次性构建脚本：生成第十二轮「连接实盘」的 4 个 execution 层文件。

运行: python tools/build_live_files.py
生成后可删除本脚本（幂等，可重复运行）。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "lianghua" / "execution"

ORDER_BOOK = '''"""订单账本：SQLite 持久化记录实盘/纸面所有订单（第十二轮·连接实盘）。

每笔经 LiveEngine 提交的订单都会 record 进 live_orders.db，可跨会话
审计与对账。零重依赖（仅标准库 sqlite3）。
"""
from __future__ import annotations

import datetime as _dt
import sqlite3


class OrderBook:
    """SQLite 订单账本。默认库文件 live_orders.db（项目根目录）。"""

    def __init__(self, db_path: str = "live_orders.db"):
        self.db_path = str(db_path)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _exec(self, sql: str, args=()):
        """执行写操作并关闭连接（Windows 下避免文件句柄锁）。"""
        con = self._conn()
        try:
            with con:
                return con.execute(sql, args)
        finally:
            con.close()

    def _init_db(self):
        self._exec(
            """CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty INTEGER NOT NULL,
                price REAL NOT NULL,
                asset TEXT DEFAULT '',
                status TEXT DEFAULT 'submitted',
                fill_price REAL DEFAULT 0.0,
                broker TEXT DEFAULT '',
                detail TEXT DEFAULT ''
            )"""
        )

    def record(self, symbol: str, side: str, qty: int, price: float,
               asset: str = "", status: str = "submitted",
               fill_price: float = 0.0, broker: str = "",
               detail: str = "") -> int:
        """记录一笔订单，返回自增 id。"""
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = self._exec(
            "INSERT INTO orders(ts,symbol,side,qty,price,asset,status,"
            "fill_price,broker,detail) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (ts, symbol, side, int(qty), float(price), asset, status,
             float(fill_price), broker, str(detail)[:500]))
        return int(cur.lastrowid)

    def update(self, oid: int, status: str | None = None,
               fill_price: float | None = None) -> None:
        sets, args = [], []
        if status is not None:
            sets.append("status=?")
            args.append(status)
        if fill_price is not None:
            sets.append("fill_price=?")
            args.append(float(fill_price))
        if not sets:
            return
        args.append(int(oid))
        self._exec("UPDATE orders SET " + ", ".join(sets) + " WHERE id=?", args)

    def _rows(self, sql: str, args=()) -> list[dict]:
        con = self._conn()
        try:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute(sql, args).fetchall()]
        finally:
            con.close()

    def pending(self) -> list[dict]:
        return self._rows("SELECT * FROM orders WHERE status='submitted' ORDER BY id")

    def recent(self, n: int = 20) -> list[dict]:
        return self._rows("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (int(n),))

    def all(self) -> list[dict]:
        return self._rows("SELECT * FROM orders ORDER BY id")

    def clear(self) -> None:
        self._exec("DELETE FROM orders")

    def __repr__(self):
        return "OrderBook(db=%r, n=%d)" % (self.db_path, len(self.all()))


__all__ = ["OrderBook"]
'''

LIVE = '''"""实盘/纸面交易执行引擎（第十二轮·连接实盘的核心胶水）。

把「策略信号 -> 盘前风控 -> broker 下单 -> 订单账本 -> 账户快照」串成
单步可重复调用的 LiveEngine：

- paper/sim broker：真实行情网关 + 模拟成交，零资金风险（默认）。
- qmt/pt broker  ：真实柜台，必须显式 live=True 才允许创建（双保险）。

用法::

    from lianghua.execution.live import make_live_engine
    eng = make_live_engine("paper", "sma_cross", ["600519.SH"], capital=1e6)
    status = eng.step()          # 执行一次「取数->信号->风控->下单」
    eng.run(max_steps=3)         # 多步循环
"""
from __future__ import annotations

import datetime as _dt
import time

import numpy as np

from ..core.assets import AssetType, detect_asset_type
from .brokers.base import Order
from .pre_trade import pre_trade_check
from .order_book import OrderBook


def _resolve_strategy(name_or_fn):
    """字符串 -> registry 策略实例；策略对象/可调用对象直接返回。"""
    if not isinstance(name_or_fn, str) and (
            hasattr(name_or_fn, "generate_signals") or callable(name_or_fn)):
        return name_or_fn
    from ..strategy.registry import get_strategy
    return get_strategy(name_or_fn)


class LiveEngine:
    """单引擎多标的实盘/纸面执行器（等权目标仓位，信号驱动调仓）。"""

    LOT = {AssetType.STOCK: 100, AssetType.FUND: 100,
           AssetType.FUTURE: 1, AssetType.OPTION: 1}

    def __init__(self, broker, gateway, strategy, symbols,
                 capital: float = 1_000_000.0, risk_limits: dict | None = None,
                 mode: str = "paper", lookback: int = 120, notify=None,
                 order_db: str = "live_orders.db", live: bool = False,
                 asset_map: dict | None = None):
        self.broker = broker
        self.gateway = gateway
        self.strategy = _resolve_strategy(strategy)
        self.symbols = list(symbols)
        self.capital = float(capital)
        self.risk_limits = dict(risk_limits or {})
        self.mode = mode
        self.lookback = int(lookback)
        self.notify = notify
        self.book = OrderBook(order_db)
        self.live = bool(live)
        self.asset_map = dict(asset_map or {})
        self.killed = False
        self.last_status: dict | None = None
        self._check_live_safety()

    # ---------- 安全 ----------
    def _check_live_safety(self):
        """真实柜台双保险：QmtBroker/PtBroker 或 mode=live 必须显式 live=True。"""
        is_real = type(self.broker).__name__ in ("QmtBroker", "PtBroker")
        if (self.mode == "live" or is_real) and not self.live:
            raise RuntimeError(
                "真实柜台/实盘模式需要显式 live=True（默认拒绝，防误触真实资金）。"
                "请确认账户、风控与策略后再传 live=True。")

    # ---------- 数据 / 信号 ----------
    def _asset(self, sym: str) -> AssetType:
        a = self.asset_map.get(sym)
        if a is not None:
            return a if isinstance(a, AssetType) else AssetType(a)
        return detect_asset_type(sym)

    def _recent(self, sym: str):
        at = self._asset(sym)
        end = _dt.date.today()
        start = end - _dt.timedelta(days=self.lookback * 2)
        df = self.gateway.fetch(sym, str(start), str(end), freq="daily",
                                asset=at.value if hasattr(at, "value") else at)
        return df, at

    def _signal(self, sym: str):
        df, at = self._recent(sym)
        sig = self.strategy.generate_signals(df)
        last = int(np.sign(float(sig.iloc[-1]))) if len(sig) else 0
        last_close = float(df["close"].iloc[-1]) if len(df) else 0.0
        return last, at, last_close, df

    # ---------- 仓位 ----------
    def _target_shares(self, at: AssetType, price: float,
                       signal: int, budget: float) -> int:
        if signal == 0 or price <= 0:
            return 0
        if signal < 0 and at not in (AssetType.FUTURE, AssetType.OPTION):
            return 0  # 股票/基金不裸卖空
        lot = self.LOT.get(at, 100)
        raw = int(budget / price / lot) * lot
        return raw if signal > 0 else -raw

    # ---------- 主循环 ----------
    def step(self) -> dict:
        """一次完整调仓：取数 -> 信号 -> 风控 -> 下单 -> 账本 -> 快照。"""
        if self.killed:
            return {"mode": self.mode, "killed": True, "actions": [],
                    "account": {}, "orders": [], "ts": str(_dt.datetime.now())}
        self.broker.connect()
        actions: list[dict] = []
        quotes: dict[str, float] = {}
        signals: dict[str, tuple] = {}
        for sym in self.symbols:
            try:
                sg, at, px, _ = self._signal(sym)
                signals[sym] = (sg, at, px)
                quotes[sym] = px
            except Exception as e:  # 单标的失败不拖垮整体
                actions.append({"symbol": sym, "action": "skip",
                                "reason": "数据/信号失败: %s" % e})
        mv = {}
        pos_now = getattr(self.broker, "positions", {}) or {}
        for s, p in pos_now.items():
            mv[s] = abs(getattr(p, "qty", 0)) * quotes.get(s, getattr(p, "avg_price", 0.0))
        acct = self.broker.get_account(mv)
        n_active = max(sum(1 for v in signals.values() if v[0] != 0), 1)
        budget = float(acct.get("cash", self.capital)) / n_active
        for sym, (sg, at, px) in signals.items():
            cur = acct.get("positions", {}).get(sym, 0)
            tgt = self._target_shares(at, px, sg, budget)
            delta = tgt - cur
            if delta == 0:
                actions.append({"symbol": sym, "action": "hold",
                                "signal": sg, "qty": cur, "price": px})
                continue
            side = "BUY" if delta > 0 else "SELL"
            order = Order(sym, side, abs(int(delta)), px, asset_type=at)
            ok, reason = pre_trade_check(order, acct, self.risk_limits)
            if not ok:
                actions.append({"symbol": sym, "action": "reject",
                                "signal": sg, "side": side,
                                "qty": abs(delta), "price": px, "reason": reason})
                continue
            try:
                res = self.broker.submit(order)
                fp = float(res.get("fill_price", px) or px)
                oid = self.book.record(sym, side, abs(int(delta)), px,
                                       asset=at.value,
                                       status=str(res.get("status", "filled")),
                                       fill_price=fp,
                                       broker=type(self.broker).__name__,
                                       detail=str(res)[:300])
                actions.append({"symbol": sym, "action": "trade", "signal": sg,
                                "side": side, "qty": abs(delta), "price": px,
                                "fill_price": fp, "order_id": oid})
                if self.notify:
                    try:
                        self.notify("[live] %s %s %d @ %.3f" % (sym, side, abs(delta), px))
                    except Exception:
                        pass
            except Exception as e:
                actions.append({"symbol": sym, "action": "error", "side": side,
                                "qty": abs(delta), "price": px, "reason": str(e)})
        # 复算最新账户
        mv2 = {}
        for s, p in (getattr(self.broker, "positions", {}) or {}).items():
            mv2[s] = abs(getattr(p, "qty", 0)) * quotes.get(s, getattr(p, "avg_price", 0.0))
        acct2 = self.broker.get_account(mv2)
        self.last_status = self._status(quotes, acct2, actions)
        return self.last_status

    def _status(self, quotes: dict, acct: dict, actions: list) -> dict:
        return {"mode": self.mode, "killed": self.killed,
                "ts": str(_dt.datetime.now()), "quotes": quotes,
                "account": acct, "actions": actions,
                "orders": self.book.recent(10)}

    def run(self, max_steps: int = 1, interval: float = 0.0) -> list[dict]:
        out = []
        for _ in range(int(max_steps)):
            if self.killed:
                break
            out.append(self.step())
            if interval > 0:
                time.sleep(interval)
        return out

    def kill(self) -> dict:
        self.killed = True
        return {"killed": True}

    def status(self) -> dict:
        return self.last_status or {"mode": self.mode, "killed": self.killed,
                                    "actions": [], "account": {}, "orders": []}

    def __repr__(self):
        return ("LiveEngine(broker=%s, mode=%s, symbols=%s, live=%s)" %
                (type(self.broker).__name__, self.mode, self.symbols, self.live))


def make_live_engine(broker_kind: str = "paper", strategy="sma_cross",
                     symbols=("600519.SH",), capital: float = 1_000_000.0,
                     gateway=None, risk_limits: dict | None = None,
                     lookback: int = 120, live: bool = False, **broker_cfg):
    """工厂：一行创建实盘引擎。qmt/pt 需 live=True（真实柜台双保险）。"""
    from .brokers import make_broker
    if gateway is None:
        from ..data.gateway import DataGateway
        gateway = DataGateway()
    is_real = broker_kind in ("qmt", "pt")
    if is_real and not live:
        raise RuntimeError(
            "broker=%s 是真实柜台，必须显式 live=True 才能创建引擎。" % broker_kind)
    kw = dict(broker_cfg)
    if broker_kind in ("sim", "paper"):
        kw.setdefault("init_cash", capital)
    broker = make_broker(broker_kind, **kw)
    return LiveEngine(broker, gateway, strategy, list(symbols), capital=capital,
                      risk_limits=risk_limits,
                      mode=("live" if is_real else "paper"),
                      lookback=lookback, live=(live or is_real))


__all__ = ["LiveEngine", "make_live_engine"]
'''

QMT = '''"""迅投 QMT (MiniQMT/xtquant) 实盘适配器（第十二轮·连接实盘）。

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
'''

PT = '''"""恒生 PTrade 实盘适配器（第十二轮·连接实盘）。

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
'''


def main():
    files = {
        ROOT / "order_book.py": ORDER_BOOK,
        ROOT / "live.py": LIVE,
        ROOT / "brokers" / "qmt.py": QMT,
        ROOT / "brokers" / "pt.py": PT,
    }
    for p, txt in files.items():
        p.write_text(txt, encoding="utf-8")
        print("WROTE", p, len(txt))


if __name__ == "__main__":
    main()
