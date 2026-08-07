"""订单账本：SQLite 持久化记录实盘/纸面所有订单（第十二轮·连接实盘）。

每笔经 LiveEngine 提交的订单都会 record 进 live_orders.db，可跨会话
审计与对账。零重依赖（仅标准库 sqlite3）。

账本是对账与绩效统计的唯一事实来源，所以数字入库前必须是有限的：原实现
校验了 price 却漏了 fill_price，柜台回报一条 NaN 成交价即可直接落库，
之后整段盈亏统计都会被这一行悄悄污染成 NaN。
"""
from __future__ import annotations

import datetime as _dt
import sqlite3

from ..core.numeric import is_finite_num


def _finite_fill(value) -> float:
    """成交价必须有限（0.0 表示"尚未成交"，合法）。"""
    if not is_finite_num(value):
        raise ValueError(f"fill_price 必须为有限数，收到 {value!r}")
    return float(value)


class OrderBook:
    """SQLite 订单账本。默认库文件 live_orders.db（项目根目录）。"""

    VALID_SIDES = ("BUY", "SELL")

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
        """记录一笔订单，返回自增 id。对 side/qty/price 做合法性守卫。"""
        side = str(side).upper()
        if side not in self.VALID_SIDES:
            raise ValueError(f"side 必须为 BUY/SELL，收到 {side!r}")
        qty = int(qty)
        if qty <= 0:
            raise ValueError(f"qty 必须为正整数，收到 {qty}")
        price = float(price)
        if not (price > 0):
            raise ValueError(f"price 必须为正数，收到 {price}")
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = self._exec(
            "INSERT INTO orders(ts,symbol,side,qty,price,asset,status,"
            "fill_price,broker,detail) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (ts, symbol, side, qty, price, asset, status,
             _finite_fill(fill_price), broker, str(detail)[:500]))
        return int(cur.lastrowid)

    def update(self, oid: int, status: str | None = None,
               fill_price: float | None = None) -> None:
        sets, args = [], []
        if status is not None:
            sets.append("status=?")
            args.append(status)
        if fill_price is not None:
            sets.append("fill_price=?")
            args.append(_finite_fill(fill_price))
        if not sets:
            return
        args.append(int(oid))
        self._exec("UPDATE orders SET " + ", ".join(sets) + " WHERE id=?", args)

    def fill(self, oid: int, fill_price: float) -> None:
        """便捷方法：将一笔订单标记为已成交流水（filled）。"""
        self.update(oid, status="filled", fill_price=float(fill_price))

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

    def summary(self) -> dict:
        """按状态汇总订单数，便于对账/审计（新增能力）。"""
        rows = self._rows(
            "SELECT status, COUNT(*) AS c FROM orders GROUP BY status")
        return {r["status"]: int(r["c"]) for r in rows}

    def clear(self) -> None:
        self._exec("DELETE FROM orders")

    def __repr__(self):
        return "OrderBook(db=%r, n=%d)" % (self.db_path, len(self.all()))


__all__ = ["OrderBook"]
