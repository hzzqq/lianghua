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
        date 统一规范为 ``%Y-%m-%d`` 字符串，避免上游传入 Timestamp/带时分秒
        的字符串导致与行情表(gateway 返回 ``%Y-%m-%d``) join 不上（隐性错位）。
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
            try:
                dt = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"特征 date 非法: {row['date']!r}") from exc
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

    def align(self, df: pd.DataFrame, symbol: str, names: list | None = None) -> pd.DataFrame:
        """把缓存特征按 date 左连接合并到 df（df 需含 date 列），返回合并后的 DataFrame。

        用于把离线/滚动特征拼回价格序列做回测或训练。date 统一规范为 ``%Y-%m-%d``
        再做 join，规避两侧日期格式不一致导致的漏连（隐性对齐陷阱）。无特征时原样返回。
        """
        if not isinstance(df, pd.DataFrame) or "date" not in df.columns:
            raise ValueError("align 需要含 date 列的 DataFrame")
        feats = self.get_features(symbol, names)
        if feats.empty:
            return df.copy()
        join = feats.copy()
        join.index = pd.to_datetime(join.index).strftime("%Y-%m-%d")
        base = df.copy()
        base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
        merged = base.merge(join, left_on="date", right_index=True, how="left")
        return merged

    def overview(self, symbol: str | None = None) -> dict:
        """可观测性：返回特征覆盖概况。

        - symbol=None：``{symbol: {features, dates, rows}}`` 全量概览；
        - 指定 symbol：``{feature_name: date_count}`` 便于排查特征缺失。
        查询失败（如表不存在）返回空 dict 而非抛错，保证内省不阻断主流程。
        """
        with sqlite3.connect(self.db) as con:
            if symbol is None:
                rows = con.execute(
                    "SELECT symbol, COUNT(DISTINCT name), COUNT(DISTINCT date), "
                    "COUNT(*) FROM feat GROUP BY symbol"
                ).fetchall()
                return {r[0]: {"features": r[1], "dates": r[2], "rows": r[3]} for r in rows}
            rows = con.execute(
                "SELECT name, COUNT(DISTINCT date) FROM feat WHERE symbol=? GROUP BY name",
                (symbol,),
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def purge(self, symbol: str | None = None):
        """清空缓存：symbol 为 None 时清空全部。"""
        with sqlite3.connect(self.db) as con:
            if symbol is None:
                con.execute("DELETE FROM feat")
            else:
                con.execute("DELETE FROM feat WHERE symbol=?", (symbol,))
