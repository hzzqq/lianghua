"""特征存储：滚动特征计算并缓存到 SQLite。"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

__all__ = ["FeatureStore"]

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS feat ("
    "symbol TEXT, date TEXT, name TEXT, value REAL, "
    "PRIMARY KEY (symbol, date, name))"
)


class FeatureStore:
    def __init__(self, db: str = "features.db"):
        self.db = db
        with sqlite3.connect(self.db) as con:
            con.execute(_SCHEMA)

    def add(self, symbol: str, df: pd.DataFrame, names: list):
        """df 需含 date 列与 names 指定的特征列；写入库。

        数据质量守卫：非有限(NaN/inf)/非数值特征会被拒绝，避免污染缓存。
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df 必须为非空 DataFrame")
        if not names:
            raise ValueError("names 不能为空")
        missing = [c for c in ("date", *names) if c not in df.columns]
        if missing:
            raise ValueError(f"df 缺少必要列: {missing}")
        df = df.copy()
        rows = []
        for _, row in df.iterrows():
            dt = str(row["date"])
            for nm in names:
                v = row[nm]
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    raise ValueError(f"特征 {nm} 在 {dt} 含非数值: {v!r}")
                if not np.isfinite(fv):
                    raise ValueError(f"特征 {nm} 在 {dt} 为非有限值(NaN/inf)，拒绝写入")
                rows.append((symbol, dt, nm, fv))
        with sqlite3.connect(self.db) as con:
            con.executemany(
                "INSERT OR REPLACE INTO feat VALUES (?,?,?,?)", rows
            )

    def get(self, symbol: str, name: str) -> pd.Series:
        with sqlite3.connect(self.db) as con:
            d = pd.read_sql_query(
                "SELECT date,value FROM feat WHERE symbol=? AND name=? ORDER BY date",
                con, params=(symbol, name),
            )
        if d.empty:
            return pd.Series(dtype=float, name=name)
        return d.set_index("date")["value"].rename(name)

    def get_features(self, symbol: str, names: list | None = None) -> pd.DataFrame:
        """一次性取出多特征，按 date 对齐成面板 DataFrame（列为各特征名）。"""
        with sqlite3.connect(self.db) as con:
            if names:
                if len(names) == 0:
                    return pd.DataFrame()
                q = "SELECT date,name,value FROM feat WHERE symbol=? AND name IN (%s)" % ",".join("?" * len(names))
                d = pd.read_sql_query(q, con, params=(symbol, *names))
            else:
                d = pd.read_sql_query(
                    "SELECT date,name,value FROM feat WHERE symbol=?",
                    con, params=(symbol,),
                )
        if d.empty:
            return pd.DataFrame()
        return d.pivot(index="date", columns="name", values="value").sort_index()

    def has(self, symbol: str, name: str) -> bool:
        """该 symbol 的该特征是否已缓存（可观测性）。"""
        with sqlite3.connect(self.db) as con:
            row = con.execute(
                "SELECT 1 FROM feat WHERE symbol=? AND name=? LIMIT 1",
                (symbol, name),
            ).fetchone()
        return row is not None

    def purge(self, symbol: str | None = None):
        """清空缓存：symbol 为 None 时清空全部。"""
        with sqlite3.connect(self.db) as con:
            if symbol is None:
                con.execute("DELETE FROM feat")
            else:
                con.execute("DELETE FROM feat WHERE symbol=?", (symbol,))
