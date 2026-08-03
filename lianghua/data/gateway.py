"""数据网关：多源接入 + 本地缓存 + 多资产路由。

设计参考 qtrader 的 DataGateway 关注点分离：
- 统一接口 fetch(symbol, start, end, freq, asset) -> pd.DataFrame
- 数据源优先级：本地缓存 -> CSV -> AKShare -> BaoStock -> 生成演示数据
- 支持四类资产：股票 / 基金 / 期货 / 期权，各自有专属演示数据生成器
- 所有历史行情落本地 SQLite 缓存，支持离线回测
  - OHLCV 资产 -> bars 表
  - 期权（含希腊字母）-> option_bars 表
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager

import numpy as np
import pandas as pd
from pathlib import Path

from ..core.assets import AssetType, detect_asset_type, get_contract_spec

logger = logging.getLogger(__name__)

CACHE_DB = Path(__file__).resolve().parents[2] / "data_cache.db"


class DataGateway:
    """行情数据网关，统一对外提供 DataFrame。"""

    COLUMNS = ["date", "open", "high", "low", "close", "volume"]
    OPTION_COLUMNS = ["date", "underlying", "option_price", "delta", "gamma",
                      "vega", "theta", "rho", "strike", "expiry", "type"]

    def __init__(self, cache_db: str | os.PathLike = CACHE_DB,
                 csv_dir: str | os.PathLike | None = None):
        self.cache_db = str(cache_db)
        self.csv_dir = Path(csv_dir) if csv_dir else None
        # 可观测性：记录最近若干次数据源失败原因，便于排障而非静默降级
        self._last_errors: list[str] = []
        # 最近一次 fetch 的来源与是否最终落到演示（假）数据，供 UI 主动告警
        self.last_source: str | None = None
        self.last_was_demo: bool = False
        # 因超时被放弃、但仍在后台真实跑着的取数线程（按数据源名分组）。
        # 用于阻止对已卡死的源继续放大请求，详见 _with_timeout / _abandoned_alive。
        self._abandoned: dict[str, list[threading.Thread]] = {}
        self._abandoned_lock = threading.Lock()
        self._init_cache()

    # ---------- 缓存 ----------
    @contextmanager
    def _connect(self):
        """打开缓存库：提交事务并**关闭**连接。

        注意 ``with sqlite3.connect(...) as con`` 只负责提交/回滚事务，并不会关闭连接。
        直接那样写会让每一次读写都泄漏一个连接句柄：长期运行的实盘进程 fd 只增不减，
        Windows 上还会把 data_cache.db 一直占住（删除/搬迁缓存库时报 WinError 32）。
        """
        con = sqlite3.connect(self.cache_db)
        try:
            with con:
                yield con
        finally:
            con.close()

    def _init_cache(self):
        with self._connect() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS bars (
                    symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
                    close REAL, volume REAL, source TEXT,
                    PRIMARY KEY (symbol, date))"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS option_bars (
                    symbol TEXT, date TEXT, underlying REAL, option_price REAL,
                    delta REAL, gamma REAL, vega REAL, theta REAL, rho REAL,
                    strike REAL, expiry TEXT, type TEXT, source TEXT,
                    PRIMARY KEY (symbol, date))"""
            )
            # 记录每个标的"已向真实源问过、并已完整落库"的日期区间。
            # 不能靠数据首尾日期反推覆盖范围：非交易日不会有行，
            # 请求 2024-01-01~2024-03-31 时数据只到 2024-01-02~2024-03-29，
            # 反推法会永远判定"未覆盖"，缓存对整月/整季度请求彻底失效。
            con.execute(
                """CREATE TABLE IF NOT EXISTS cache_range (
                    symbol TEXT, tbl TEXT, start TEXT, end TEXT,
                    PRIMARY KEY (symbol, tbl))"""
            )
            # 老缓存库兼容：缺 source 列则补上
            for tbl in ("bars", "option_bars"):
                try:
                    con.execute(f"ALTER TABLE {tbl} ADD COLUMN source TEXT")
                except Exception:
                    pass

    def _cache_get(self, symbol: str, start: str, end: str, asset: AssetType) -> pd.DataFrame | None:
        try:
            table = "option_bars" if asset == AssetType.OPTION else "bars"
            cols = self.OPTION_COLUMNS if asset == AssetType.OPTION else self.COLUMNS
            with self._connect() as con:
                df = pd.read_sql_query(
                    f"SELECT * FROM {table} WHERE symbol=? AND date>=? AND date<=? ORDER BY date",
                    con,
                    params=(symbol, start, end),
                )
            if df.empty:
                return None
            cols = list(cols) + ["source"]
            out = df[[c for c in cols if c in df.columns]]
            # 演示(假)数据绝不应从缓存中继续提供：一旦命中 demo 缓存就当作未命中，
            # 迫使本次 fetch 重新尝试真实源，避免"首次失败→假数据被永久缓存→用户永远看到假数据"
            if "source" in out.columns and (out["source"] == "demo").any():
                return None
            return out
        except Exception:
            return None

    def _cache_put(self, symbol: str, df: pd.DataFrame, asset: AssetType, source: str = "unknown",
                   start: str | None = None, end: str | None = None):
        try:
            df = df.copy()
            df["symbol"] = symbol
            df["source"] = source
            table = "option_bars" if asset == AssetType.OPTION else "bars"
            # 本次覆盖区间 = 向真实源问到的窗口 ∪ 实际拿回的数据首尾。
            # 请求窗口本身就是"已问过"的范围，哪怕其中某些日子没有行（休市），
            # 也属于已知信息，不该再为此重复打网络。
            data_lo = str(df["date"].min()) if not df.empty else None
            data_hi = str(df["date"].max()) if not df.empty else None
            lo = hi = None
            if start and end and data_lo:
                lo = min(start, data_lo)
                # 尚未收盘/尚未产生的日期不能算"已问过"：盘中取一次
                # [start, 今天] 后，源里还没有今天的K线，若把覆盖区间记到今天，
                # 收盘后再取会一直命中缓存、永远看不到当天行情（实盘会用昨天的价格做决策）。
                # 历史区间不会再变，可以放心声称覆盖。
                hi = min(max(end, data_hi), self._last_settled_date())
                if hi < lo:  # 只问了当天：当天数据始终重新拉取，不进覆盖区间
                    lo = hi = None
            with self._connect() as con:
                old = None
                if lo:
                    old = con.execute(
                        "SELECT start, end FROM cache_range WHERE symbol=? AND tbl=?",
                        (symbol, table),
                    ).fetchone()
                if lo and old and old[0] and old[1] and self._ranges_join(old[0], old[1], lo, hi):
                    # 新旧窗口相交或首尾相邻：只覆盖重叠的那段，旧区间的数据继续保留。
                    # 否则用户在 UI 上来回切时间区间（Q1→Q2→回看 Q1）会把上一段缓存
                    # 整块删掉，每切一次都重新打网络。
                    #
                    # 删除范围必须覆盖本次要写入的全部行（data_lo~data_hi），而不只是
                    # 可声称的覆盖区间：当天数据不进覆盖区间，但它确实要落库，
                    # 漏删就会撞主键、让整次写入回滚，缓存从此再也刷不新。
                    con.execute(
                        f"DELETE FROM {table} WHERE symbol=? AND date>=? AND date<=?",
                        (symbol, min(lo, data_lo), max(hi, data_hi)),
                    )
                    lo, hi = min(lo, str(old[0])), max(hi, str(old[1]))
                else:
                    # 与已有区间之间存在缺口（或本次没有区间信息）：整块替换。
                    # 保留缺口数据会让 cache_range 谎称覆盖了从未拉取过的日期。
                    con.execute(f"DELETE FROM {table} WHERE symbol=?", (symbol,))
                df.to_sql(table, con, if_exists="append", index=False)
                if lo:
                    con.execute(
                        "INSERT OR REPLACE INTO cache_range(symbol, tbl, start, end) VALUES (?,?,?,?)",
                        (symbol, table, lo, hi),
                    )
        except Exception as exc:  # noqa: BLE001
            # 缓存写失败不该中断取数，但必须留痕：静默吞掉会让"缓存永远刷不新"
            # 这类问题彻底不可见（本次即由此暴露）。
            logger.warning("行情缓存写入失败 %s/%s: %s", symbol, asset, exc)

    @staticmethod
    def _last_settled_date() -> str:
        """最后一个"数据不会再变"的自然日 = 昨天。

        今天的行情盘中随时在变、收盘后才定稿，把今天算进缓存覆盖区间
        会让当天的后续请求全部命中陈旧缓存。
        """
        return (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    @staticmethod
    def _ranges_join(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
        """两个日期区间相交或首尾相邻（合并后中间没有缺口）时为 True。

        相邻按自然日判断：[01-01, 03-31] 与 [04-01, 06-30] 合并后是连续的
        [01-01, 06-30]，中间没有任何未拉取过的日期，可以安全合并。
        """
        try:
            one = pd.Timedelta(days=1)
            a0, a1 = pd.Timestamp(a_start), pd.Timestamp(a_end)
            b0, b1 = pd.Timestamp(b_start), pd.Timestamp(b_end)
        except Exception:  # noqa: BLE001
            return False
        return a0 <= b1 + one and b0 <= a1 + one

    def _cache_covers(self, symbol: str, asset: AssetType, start: str, end: str,
                      cached: pd.DataFrame) -> bool:
        """判断本地缓存是否已完整覆盖 [start, end]。"""
        table = "option_bars" if asset == AssetType.OPTION else "bars"
        try:
            with self._connect() as con:
                row = con.execute(
                    "SELECT start, end FROM cache_range WHERE symbol=? AND tbl=?",
                    (symbol, table),
                ).fetchone()
        except Exception:
            row = None
        if row and row[0] and row[1]:
            return str(row[0]) <= start and str(row[1]) >= end
        # 老缓存库没有覆盖区间记录：退回按数据首尾日期判断。
        # 这条路偏保守（非交易日边界会误判未覆盖而重新拉取），但不会返回缺口数据。
        return str(cached["date"].min()) <= start and str(cached["date"].max()) >= end

    # ---------- 数据源：CSV ----------
    def _from_csv(self, symbol: str) -> pd.DataFrame | None:
        if not self.csv_dir:
            return None
        for cand in (f"{symbol}.csv", f"{symbol.replace('.', '_')}.csv"):
            p = self.csv_dir / cand
            if p.exists():
                df = pd.read_csv(p)
                return self._normalize(df)
        return None

    # ---------- 数据源：AKShare（多资产） ----------
    def _from_akshare(self, symbol: str, start: str, end: str, asset: AssetType) -> pd.DataFrame | None:
        try:
            import akshare as ak  # 延迟导入，缺包不影响核心逻辑
        except Exception:
            return None
        try:
            if asset == AssetType.FUND:
                return self._ak_fund(ak, symbol, start, end)
            if asset == AssetType.FUTURE:
                return self._ak_future(ak, symbol, start, end)
            if asset == AssetType.OPTION:
                return self._ak_option(ak, symbol, start, end)
            return self._ak_stock(ak, symbol, start, end)
        except Exception:
            # 网络不可用 / 接口变更，回退到演示数据
            return None

    @staticmethod
    def _ak_stock(ak, symbol, start, end):
        code = symbol.split(".")[0]
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.replace("-", ""), end_date=end.replace("-", ""),
            adjust="qfq",
        )
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
        })
        return DataGateway._normalize(df)

    @staticmethod
    def _ak_fund(ak, symbol, start, end):
        # 先试 ETF 日线（代码多为 5/1 开头，如 510300），失败再试开放式基金净值
        code = symbol.split(".")[0]
        start_s, end_s = start.replace("-", ""), end.replace("-", "")
        try:
            df = ak.fund_etf_hist_em(
                symbol=code, period="daily",
                start_date=start_s, end_date=end_s, adjust="qfq",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                })
                return DataGateway._normalize(df)
        except Exception:
            pass
        try:
            df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            if df is not None and not df.empty:
                df = df.rename(columns={"净值日期": "date", "单位净值": "close"})
                df["open"] = df["high"] = df["low"] = df["close"]
                df["volume"] = 0.0
                return DataGateway._normalize(df[(df["date"] >= start) & (df["date"] <= end)])
        except Exception:
            return None
        return None

    @staticmethod
    def _ak_future(ak, symbol, start, end):
        # 主连日线（去交易所后缀取品种代码 + 主连 "0"）
        code = symbol.split(".")[0]
        variety = "".join(filter(str.isalpha, code)) + "0"
        df = ak.futures_main_sina(symbol=variety)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return DataGateway._normalize(df[(df["date"] >= start) & (df["date"] <= end)])

    @staticmethod
    def _ak_option(ak, symbol, start, end):
        """期权真实源（可用接口，替代已失效的 option_sina_sse_hfq）：

        1) 优先真实期权日线 ``option_sse_daily_sina``（真实期权价）；
        2) 退化：真实标的 ETF 日线（``fund_etf_hist_em``）+ BS 模型期权价
           —— 仅标的为真实，期权价与希腊字母均由 BS 模型合成（非真实盘口）；
           该分支会在返回的 DataFrame.attrs 标注 ``synthetic_options=True``，
           由 ``fetch`` 按合成(类 demo)数据处置（不落缓存、如实标注、主动告警）。
        3) 任一失败返回 None，由 ``fetch`` 降级到演示数据。

        返回列对齐 ``OPTION_COLUMNS``；分支 1（真实期权盘口）标记 source=akshare，
        分支 2（BS 合成）标记 synthetic_options 由 fetch 降级为合成(类 demo)处置。
        """
        from ..option.pricing import bs_price, greeks
        code = symbol.split(".")[0]
        underlying_code = code[:6]            # 510050C3000 -> 510050（50ETF）
        start_s, end_s = start.replace("-", ""), end.replace("-", "")
        info = get_contract_spec(symbol, AssetType.OPTION)
        K, otype = info.strike, info.opt_type
        r, sigma = 0.02, 0.20
        expiry = pd.Timestamp(end) + pd.Timedelta(days=90)

        # 1) 真实期权日线
        try:
            df = ak.option_sse_daily_sina(symbol=code)
            if df is not None and not df.empty and "收盘" in df.columns:
                df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                        "最低": "low", "收盘": "close", "成交量": "volume"})
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date")
                if not df.empty:
                    udf = ak.fund_etf_hist_em(symbol=underlying_code, period="daily",
                                              start_date=start_s, end_date=end_s, adjust="qfq")
                    if udf is not None and not udf.empty:
                        udf = udf.rename(columns={"日期": "date", "收盘": "close"})
                        udf["date"] = pd.to_datetime(udf["date"]).dt.strftime("%Y-%m-%d")
                        m = udf.set_index("date")["close"]
                        rows = []
                        for _, row in df.iterrows():
                            S = float(m.get(row["date"], np.nan))
                            if pd.isna(S):
                                S = float(row["close"]) * K / 3.0   # 兜底反推
                            T = max(0.02, (expiry - pd.Timestamp(row["date"])).days / 365.0)
                            g = greeks(S, K, T, r, sigma, otype)
                            rows.append({
                                "date": row["date"], "underlying": S,
                                "option_price": float(row["close"]),
                                "delta": g["delta"], "gamma": g["gamma"], "vega": g["vega"],
                                "theta": g["theta"], "rho": g["rho"],
                                "strike": K, "expiry": expiry.strftime("%Y-%m-%d"), "type": otype,
                            })
                        return pd.DataFrame(rows)
        except Exception:
            pass

        # 2) 真实标的 ETF 日线 + BS 模型期权价（真实标的锚定）
        try:
            udf = ak.fund_etf_hist_em(symbol=underlying_code, period="daily",
                                      start_date=start_s, end_date=end_s, adjust="qfq")
            if udf is None or udf.empty:
                return None
            udf = udf.rename(columns={"日期": "date", "收盘": "close"})
            udf["date"] = pd.to_datetime(udf["date"]).dt.strftime("%Y-%m-%d")
            udf = udf[(udf["date"] >= start) & (udf["date"] <= end)].sort_values("date")
            if udf.empty:
                return None
            rows = []
            for _, row in udf.iterrows():
                S = float(row["close"])
                T = max(0.02, (expiry - pd.Timestamp(row["date"])).days / 365.0)
                price = bs_price(S, K, T, r, sigma, otype)
                g = greeks(S, K, T, r, sigma, otype)
                rows.append({
                    "date": row["date"], "underlying": S, "option_price": price,
                    "delta": g["delta"], "gamma": g["gamma"], "vega": g["vega"],
                    "theta": g["theta"], "rho": g["rho"],
                    "strike": K, "expiry": expiry.strftime("%Y-%m-%d"), "type": otype,
                })
            # 此分支仅标的(ETF)为真实，期权价与希腊字母均由 BS 模型合成——并非真实盘口报价。
            # 标记后由 fetch 按合成(类 demo)数据处置：不冒充 akshare、不落缓存、主动告警，
            # 避免用户把模型合成期权价当真实盘口永久缓存并用于回测。
            out = pd.DataFrame(rows)
            out.attrs["synthetic_options"] = True
            return out
        except Exception:
            return None

    # ---------- 数据源：BaoStock（A 股） ----------
    def _from_baostock(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        try:
            import baostock as bs
        except Exception:
            return None
        try:
            # 代码转换：600519.SH -> sh.600519
            code, market = symbol.split(".")
            bs_code = f"{market.lower()}.{code}"
            lg = bs.login()
            if not lg or lg.error_code != "0":
                return None
            rs = bs.query_history_k_data_plus(
                bs_code, "date,open,high,low,close,volume",
                start_date=start, end_date=end, frequency="d", adjustflag="2",
            )
            if rs is None or rs.error_code != "0":
                bs.logout()
                return None
            rows = []
            while (row := rs.next()) is not None:
                rows.append(row)
            bs.logout()
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return self._normalize(df)
        except Exception:
            return None

    # ---------- 归一化 ----------
    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in DataGateway.COLUMNS:
            if c not in df.columns:
                df[c] = 0.0
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[DataGateway.COLUMNS].sort_values("date").reset_index(drop=True)
        return df

    # ---------- 多资产演示数据 ----------
    @staticmethod
    def _seed(symbol: str) -> np.random.Generator:
        import hashlib
        # 用 hashlib 而非内置 hash()，保证演示数据跨进程可复现
        seed_int = int(hashlib.md5(symbol.encode("utf-8")).hexdigest(), 16) % (2**32)
        return np.random.default_rng(seed_int)

    @staticmethod
    def _ohlc_walk(dates, seed, start=100.0, mu=0.0005, sigma=0.02):
        n = len(dates)
        ret = seed.normal(mu, sigma, n)
        close = start * (1 + ret).cumprod()
        open_ = close * (1 + seed.normal(0, 0.003, n))
        high = np.maximum(open_, close) * (1 + abs(seed.normal(0, 0.01, n)))
        low = np.minimum(open_, close) * (1 - abs(seed.normal(0, 0.01, n)))
        vol = seed.integers(1e5, 1e6, n).astype(float)
        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": open_, "high": high, "low": low,
            "close": close, "volume": vol,
        })

    @classmethod
    def _demo_stock(cls, symbol, start, end):
        dates = pd.bdate_range(start, end)
        seed = cls._seed(symbol)
        return cls._ohlc_walk(dates, seed, start=100.0)

    @classmethod
    def _demo_fund(cls, symbol, start, end):
        """基金：净值(NAV)随机游走，close 即单位净值。"""
        dates = pd.bdate_range(start, end)
        seed = cls._seed(symbol)
        nav = 1.0 * (1 + seed.normal(0.0003, 0.012, len(dates))).cumprod()
        return cls._ohlc_walk(dates, seed, start=nav[0])

    @classmethod
    def _demo_future(cls, symbol, start, end):
        """期货：与股票类似的价格路径，规格由合约元数据决定。"""
        dates = pd.bdate_range(start, end)
        seed = cls._seed(symbol)
        spec = get_contract_spec(symbol, AssetType.FUTURE)
        base = 100.0 * spec.multiplier if spec.multiplier > 50 else 3000.0
        return cls._ohlc_walk(dates, seed, start=base)

    @classmethod
    def _demo_option(cls, symbol, start, end):
        """期权：生成标的路径 + 每日 BS 定价 + 希腊字母。"""
        from ..option.pricing import bs_price, greeks
        info = get_contract_spec(symbol, AssetType.OPTION)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        seed = cls._seed(symbol)
        s0 = info.strike * (1 + seed.normal(0, 0.02))
        underlying = s0 * (1 + seed.normal(0.0004, 0.018, n)).cumprod()
        r = 0.02
        sigma = 0.22
        T = 0.25  # 约 3 个月
        rows = []
        for i, d in enumerate(dates):
            S = float(underlying[i])
            K = info.strike
            price = bs_price(S, K, T, r, sigma, info.opt_type)
            g = greeks(S, K, T, r, sigma, info.opt_type)
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "underlying": S,
                "option_price": price,
                "delta": g["delta"], "gamma": g["gamma"],
                "vega": g["vega"], "theta": g["theta"], "rho": g["rho"],
                "strike": K, "expiry": (pd.Timestamp(d) + pd.Timedelta(days=int(T * 365))).strftime("%Y-%m-%d"),
                "type": info.opt_type,
            })
        return pd.DataFrame(rows)

    def _demo_data(self, symbol: str, start: str, end: str, asset: AssetType) -> pd.DataFrame:
        if asset == AssetType.FUND:
            return self._demo_fund(symbol, start, end)
        if asset == AssetType.FUTURE:
            return self._demo_future(symbol, start, end)
        if asset == AssetType.OPTION:
            return self._demo_option(symbol, start, end)
        return self._demo_stock(symbol, start, end)

    # ---------- 对外接口 ----------
    def fetch(self, symbol: str, start: str, end: str, freq: str = "daily",
              asset: AssetType | str | None = None, timeout: float = 15.0,
              force_refresh: bool = False, retries: int = 1, backoff: float = 0.0) -> pd.DataFrame:
        """获取行情。优先 缓存 -> CSV -> AKShare -> BaoStock -> 演示数据。

        对股票/基金/期货返回 OHLCV；期权返回含希腊字母的专用结构。
        四类资产均会尝试真实源（AKShare）；网络不可用/超时/接口变更时自动降级演示数据。

        retries/backoff: 真实源（AKShare/BaoStock）在超时或异常时按 ``retries`` 次重试，
            每次间隔 ``backoff`` 秒（默认不重试、间隔 0，便于单测；线上可调大以扛瞬时抖动）。
            仅对"真实源"重试——缓存/CSV 命中与最终演示降级不重试。
        force_refresh: 为 True 时跳过缓存强制重新拉取（用于刷新/更新数据）。
        缓存仅在**完整覆盖**请求区间时才命中，避免部分缓存返回不完整数据。

        降级可观测：最终落到演示（假）数据时，会在日志打 warning 并把失败原因记入
        ``_last_errors``；调用方可读 ``last_was_demo`` / ``last_source`` 主动提示用户，
        避免用户在不经意间拿假数据跑回测/分析。
        """
        at = AssetType(asset) if isinstance(asset, str) else (asset or detect_asset_type(symbol))

        # 输入守卫：非法日期区间直接报错，而非静默返回空（隐性数据质量陷阱）
        try:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"非法的 start/end 日期: {start!r}/{end!r}") from exc
        if s > e:
            raise ValueError(f"start({start}) 不能晚于 end({end})")

        if not force_refresh:
            cached = self._cache_get(symbol, start, end, at)
            if cached is not None and not cached.empty:
                # 仅当缓存完整覆盖 [start, end] 才命中；否则回退到重新拉取补齐缺口
                if self._cache_covers(symbol, at, start, end, cached):
                    self.last_source, self.last_was_demo = "cache", False
                    return cached[(cached["date"] >= start) & (cached["date"] <= end)].reset_index(drop=True)

        df = self._from_csv(symbol)
        src = "csv" if df is not None else None
        # 全资产尝试 AKShare 真实源（带超时+重试保护，避免瞬时抖动直接降级假数据）
        if df is None:
            df = self._try_source(self._from_akshare, timeout, retries, backoff,
                                  symbol, start, end, at)
            src = "akshare" if df is not None else None
        # 股票额外尝试 BaoStock
        if df is None and at == AssetType.STOCK:
            df = self._try_source(self._from_baostock, timeout, retries, backoff,
                                  symbol, start, end)
            src = "baostock" if df is not None else None
        # 期权盘口在真实源不可用时，可能由 BS 模型依据真实标的合成期权价/希腊字母；
        # 这类合成期权价不是真实盘口报价，绝不能冒充 akshare（真实数据）落库、永久缓存，
        # 否则用户做期权回测会拿到静默错误数字。按合成(类 demo)数据处置：不落缓存 + 可观测告警。
        if (df is not None and at == AssetType.OPTION
                and getattr(df, "attrs", {}).get("synthetic_options")):
            src = "demo"
            self._last_errors.append(
                "_ak_option: 真实期权盘口不可用，期权价由 BS 模型依据真实标的合成(非真实盘口)，按合成数据处置"
            )
        if df is None or df.empty:
            df = self._demo_data(symbol, start, end, at)
            src = "demo"

        # 降级可观测：演示（假）数据命中时明确告警，避免静默误导
        self.last_source = src
        self.last_was_demo = (src == "demo")
        if src == "demo" and self._last_errors:
            logger.warning(
                "标 %s 真实行情获取失败，已降级为演示（假）数据；原因: %s",
                symbol, "; ".join(self._last_errors[-3:]),
            )

        # 真实/演示统一在返回前标注来源，保证首次 fetch 也可追溯
        df = df.copy()
        df["source"] = src or "akshare"
        # 演示(假)数据不落缓存：避免假数据被永久缓存后，后续 fetch 直接命中缓存、
        # 不再尝试真实源，导致用户永远在不知情下看到假数据。
        if src != "demo":
            self._cache_put(symbol, df, at, source=src or "akshare", start=start, end=end)
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    def _try_source(self, fn, timeout: float, retries: int, backoff: float, *args):
        """带超时与重试地执行真实源取数；全失败返回 None（由 fetch 降级演示数据）。

        真实源可能因瞬时网络抖动/接口限流而偶发失败，重试可显著减少不必要的假数据降级。
        每次失败原因都记入 ``_last_errors``，便于排障；``backoff`` 控制重试间隔（秒）。
        """
        attempts = max(1, int(retries) + 1)
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                res = self._with_timeout(fn, timeout, *args)
                if res is not None and not (isinstance(res, pd.DataFrame) and res.empty):
                    # 成功：清掉本次尝试前累计的瞬时失败记录，避免陈旧原因误导
                    self._last_errors = [m for m in self._last_errors
                                          if not m.startswith(getattr(fn, "__name__", ""))]
                    return res
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            if i < attempts - 1 and backoff > 0:
                import time
                time.sleep(backoff)
        if last_exc is not None:
            self._last_errors.append(
                f"{getattr(fn, '__name__', fn)}: 重试{attempts - 1}次仍失败: {type(last_exc).__name__}: {last_exc}"
            )
        else:
            # 真实源未抛异常但返回空（如依赖未安装/接口返回空），同样记录以便降级告警
            self._last_errors.append(
                f"{getattr(fn, '__name__', fn)}: 重试{attempts - 1}次仍无有效数据"
            )
        return None

    def _note_error(self, fn, exc: BaseException) -> None:
        """记录一次真实源失败原因（保留最近 20 条，供 UI/排障读取）。"""
        self._last_errors.append(
            f"{getattr(fn, '__name__', fn)}: {type(exc).__name__}: {exc}")
        if len(self._last_errors) > 20:
            self._last_errors = self._last_errors[-20:]

    def _with_timeout(self, fn, timeout: float, *args):
        """在独立**守护**线程中执行网络请求，超时则返回 None（降级演示数据）。

        不能用 ``with ThreadPoolExecutor(...)``：它的 ``__exit__`` 会
        ``shutdown(wait=True)`` 去 join 工作线程，于是 ``fut.result(timeout=)``
        抛出 TimeoutError 之后照样被卡在退出处 —— 超时保护形同虚设。

        实测危害：无网环境下 baostock 的 ``rs.next()`` 会阻塞十几分钟以上，
        ``fetch(timeout=15)`` 却一直不返回，既不降级也不报错：
        - ``LiveEngine.step()`` 的调仓循环整体僵死，监控看不到任何 error；
        - 全量测试从几分钟拖到一小时以上仍跑不完。

        改用 daemon 线程 + ``join(timeout)``：超时后主动放弃该线程（daemon 不会
        阻塞解释器退出），调用方在 ``timeout`` 秒内一定拿到结果或降级信号。

        放弃线程留下的后果必须自己兜住：被放弃的线程仍在真实地占着一条连接跑。
        若不管它，同一个卡死的源会被反复重新请求（``_try_source`` 的重试、
        ``LiveEngine`` 每轮轮询），于是请求被放大、守护线程与 socket 无上限堆积。
        因此这里加一道**在途守卫**：某个源上一次被放弃的线程只要还活着，就直接
        快速失败降级，不再对它发起新请求。守卫是自愈的——那条线程一旦真的返回，
        该源立刻恢复可用，不依赖任何超时冷却常量。
        """
        name = getattr(fn, "__name__", str(fn))
        if self._abandoned_alive(name):
            self._note_error(fn, TimeoutError(
                "上一次取数超时后仍未返回，本次直接降级；"
                "避免对已卡死的源放大请求并堆积线程"))
            return None

        box: dict = {}

        def _run():
            try:
                box["value"] = fn(*args)
            except BaseException as exc:  # noqa: BLE001 - 原样带回主线程记录
                box["error"] = exc

        t = threading.Thread(target=_run, daemon=True,
                             name=f"lianghua-fetch-{name}")
        t.start()
        t.join(max(0.0, float(timeout)))
        if t.is_alive():
            # 线程仍在跑：放弃等待，登记为在途后如实记录超时原因（不 join、不阻塞调用方）
            with self._abandoned_lock:
                self._abandoned.setdefault(name, []).append(t)
            self._note_error(fn, TimeoutError(f"超过 {timeout}s 未返回，已放弃本次取数"))
            return None
        if "error" in box:
            self._note_error(fn, box["error"])
            return None
        return box.get("value")

    def _abandoned_alive(self, name: str) -> int:
        """回收已结束的被放弃线程，返回该源仍在途的被放弃线程数。

        只统计**因超时被放弃**的线程：正常完成的请求线程会自然退出并在此被清掉，
        所以并发地对同一个健康源取数不会被本守卫误伤。
        """
        with self._abandoned_lock:
            alive = [t for t in self._abandoned.get(name, ()) if t.is_alive()]
            if alive:
                self._abandoned[name] = alive
            else:
                self._abandoned.pop(name, None)
            return len(alive)

    def last_warnings(self) -> list[str]:
        """返回最近的数据源失败原因（用于 UI 主动提示用户假数据降级）。"""
        return list(self._last_errors)

    # ---------- 分钟线（日内） ----------
    def fetch_minute(self, symbol: str, day: str, asset: str | None = None,
                     freq: str = "1min", bars: int = 240) -> pd.DataFrame:
        """获取某标的指定交易日的分钟线（演示数据，断网可跑）。

        返回含 open/high/low/close/volume 的 DataFrame，index 重置后含 datetime 列。
        真实接入可在内部调用 AKShare 分钟接口（如 stock_zh_a_hist_min_em）。
        """
        at = AssetType(asset) if isinstance(asset, str) else (asset or detect_asset_type(symbol))
        # 输入守卫：bars 必须为正，否则 close[0] 会越界崩溃（隐性崩溃风险）
        if not isinstance(bars, int):
            raise ValueError(f"bars 必须为正整数，收到 {bars!r}")
        if bars <= 0:
            raise ValueError(f"bars 必须为正整数，收到 {bars}")
        rng = self._seed(symbol + "|" + day)
        idx = pd.date_range(f"{day} 09:30:00", periods=bars, freq=freq)
        drift = np.linspace(0, rng.normal(0, 0.02), bars)
        noise = rng.normal(0, 0.001, bars).cumsum() * 0.01
        close = 100.0 * np.exp(drift + noise)
        high = close * (1 + np.abs(rng.normal(0, 0.0006, bars)))
        low = close * (1 - np.abs(rng.normal(0, 0.0006, bars)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        vol = rng.integers(100, 1000, bars)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=idx,
        )
        return df.reset_index().rename(columns={"index": "datetime"})

    # ---------- 缓存内省（可观测性新能力） ----------
    def list_cached(self, asset: AssetType | str | None = None) -> dict:
        """列出当前缓存中已落库的标的及其区间/来源。

        返回 ``{symbol: {"min": date, "max": date, "rows": int, "source": str}}``，
        便于运维与离线回测前确认数据覆盖情况。``asset`` 指定时只返回该资产类别的标的。

        查询失败（如表不存在）时返回空 dict 而非抛错，保证内省本身不阻断主流程。
        """
        at = AssetType(asset) if isinstance(asset, str) else asset
        # 缓存表不记录资产类别，仅能区分 OPTION（独立表）与非期权（bars 混合三类）。
        if at == AssetType.OPTION:
            tables = ("option_bars",)
        elif at is not None:
            tables = ("bars",)
        else:
            tables = ("bars", "option_bars")
        out: dict = {}
        for table in tables:
            for sym, dmin, dmax, src in self._query_all(
                f"SELECT symbol, MIN(date), MAX(date), MAX(source) FROM {table} GROUP BY symbol"
            ):
                out[sym] = {"min": dmin, "max": dmax, "rows": self._count(table, sym),
                            "source": src or "unknown"}
        return out

    def _query_all(self, sql: str):
        try:
            with self._connect() as con:
                cur = con.execute(sql)
                return cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_cached 查询失败: %s", exc)
            return []

    def _count(self, table: str, symbol: str) -> int:
        try:
            with self._connect() as con:
                return con.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE symbol=?", (symbol,)
                ).fetchone()[0]
        except Exception:  # noqa: BLE001
            return 0

    # ---------- 实时快照（连接实盘） ----------
    def live_quote(self, symbol: str, asset: str | None = None) -> dict:
        """实时快照（最新价）。优先 AKShare 实时接口，失败降级到最后收盘/演示价。

        返回 {"symbol", "price", "source"}，source ∈ {akshare, last_close, none}。
        用于实盘引擎的市值标记与决策。
        """
        at = AssetType(asset) if isinstance(asset, str) else (asset or detect_asset_type(symbol))
        try:
            import akshare as ak
            code = symbol.split(".")[0]
            if at == AssetType.FUND:
                df = ak.fund_etf_spot_em()
                row = df[df["代码"] == code]
            else:
                df = ak.stock_zh_a_spot_em()
                row = df[df["代码"] == code]
            if not row.empty:
                return {"symbol": symbol,
                        "price": float(row.iloc[0]["最新价"]),
                        "source": "akshare"}
        except Exception:
            pass
        # 降级：最近日线收盘
        end = pd.Timestamp.now().strftime("%Y-%m-%d")
        start = (pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        try:
            df = self.fetch(symbol, start, end, asset=at)
            if df is not None and not df.empty:
                # 若取数被静默降级为演示(假)数据，必须如实标注 source=demo，
                # 绝不能冒充真实收盘价(last_close)，否则实盘引擎会用假价格做市值标记。
                src = "demo" if str(df["source"].iloc[-1]) == "demo" else "last_close"
                self.last_source, self.last_was_demo = src, (src == "demo")
                return {"symbol": symbol, "price": float(df["close"].iloc[-1]),
                        "source": src}
        except Exception:
            pass
        return {"symbol": symbol, "price": float("nan"), "source": "none"}
