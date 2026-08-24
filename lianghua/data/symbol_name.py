"""标的代码 ↔ 中文名称映射（迭代 181+ 超目标）。

提供 ``get_symbol_name(symbol, asset=None)`` 与 ``get_symbols_names(symbols)``，
用于 UI 在代码输入框旁实时显示中文名称。查询优先级：

1. 本地内置映射（离线可用，覆盖常见 A 股 / 基金 / 期货）
2. AKShare 在线查询（网络可用时自动补充）
3. 规则化兜底（如期货品种名 + 合约月、期权底层 + 购/沽）

结果按 symbol 缓存，避免重复网络请求拖慢 UI 交互。
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from pypinyin import lazy_pinyin
    _HAS_PYPINYIN = True
except Exception:  # pragma: no cover - 无 pypinyin 时首字母搜索失效但程序不崩
    lazy_pinyin = None  # type: ignore
    _HAS_PYPINYIN = False

# 缓存：symbol -> name
_CACHE: dict[str, str] = {}

# ---------- 内置名称映射（离线兜底） ----------
_STOCK_NAMES: dict[str, str] = {
    # 白酒
    "600519.SH": "贵州茅台", "000858.SZ": "五粮液", "000568.SZ": "泸州老窖",
    "002304.SZ": "洋河股份", "600779.SH": "水井坊", "600809.SH": "山西汾酒",
    # 银行
    "601398.SH": "工商银行", "601318.SH": "中国平安", "600036.SH": "招商银行",
    "601166.SH": "兴业银行", "600000.SH": "浦发银行", "601328.SH": "交通银行",
    "601988.SH": "中国银行", "002142.SZ": "宁波银行",
    # 新能源 / 科技
    "300750.SZ": "宁德时代", "002594.SZ": "比亚迪", "601012.SH": "隆基绿能",
    "600438.SH": "通威股份", "300014.SZ": "亿纬锂能", "002415.SZ": "海康威视",
    "000725.SZ": "京东方A", "002230.SZ": "科大讯飞", "688981.SH": "中芯国际",
    "000063.SZ": "中兴通讯", "600524.SH": "岭南控股", "002371.SZ": "北方华创",
    "603501.SH": "韦尔股份", "600745.SH": "闻泰科技", "002049.SZ": "紫光国微",
    "000100.SZ": "TCL科技", "600460.SH": "士兰微",
    # 医药
    "600276.SH": "恒瑞医药", "300760.SZ": "迈瑞医疗", "000661.SZ": "长春高新",
    "600196.SH": "复星医药", "002821.SZ": "凯莱英", "600436.SH": "片仔癀",
    "000538.SZ": "云南白药", "603259.SH": "药明康德", "300142.SZ": "沃森生物",
    # 上证50 / 沪深300 龙头
    "600900.SH": "长江电力", "601888.SH": "中国中免", "601628.SH": "中国人寿",
    "600030.SH": "中信证券", "600837.SH": "海通证券", "000651.SZ": "格力电器",
    "002475.SZ": "立讯精密", "600585.SH": "海螺水泥", "000625.SZ": "长安汽车",
    "600999.SH": "招商证券", "000001.SZ": "平安银行", "601668.SH": "中国建筑",
    "601899.SH": "紫金矿业", "600028.SH": "中国石化", "601088.SH": "中国神华",
    "601857.SH": "中国石油", "600009.SH": "上海机场", "601111.SH": "中国国航",
    "600048.SH": "保利发展", "000002.SZ": "万科A",
    # 消费 / 制造
    "600887.SH": "伊利股份", "002714.SZ": "牧原股份", "300498.SZ": "温氏股份",
    "000895.SZ": "双汇发展", "603288.SH": "海天味业", "600132.SH": "重庆啤酒",
    "000568.SZ": "泸州老窖", "000860.SZ": "顺鑫农业",
    # 汽车 / 机械
    "000768.SZ": "中航西飞", "600760.SH": "中航沈飞", "002236.SZ": "大华股份",
    "002594.SZ": "比亚迪", "601127.SH": "赛力斯", "002466.SZ": "天齐锂业",
    "603799.SH": "华友钴业", "300073.SZ": "当升科技", "002074.SZ": "国轩高科",
    # 图例：拼音首字母搜索 hmd 示例
    "600510.SH": "黑牡丹", "002947.SZ": "恒铭达", "301682.SZ": "宏明电子",
    # 其它常见
    "000300.SH": "沪深300指数", "000016.SH": "上证50指数", "000905.SH": "中证500指数",
    "399006.SZ": "创业板指数",
}

_FUND_NAMES: dict[str, str] = {
    "510300.SH": "沪深300ETF", "510050.SH": "上证50ETF", "510500.SH": "中证500ETF",
    "159915.SZ": "创业板ETF", "159901.SZ": "深100ETF", "512000.SH": "券商ETF",
    "512010.SH": "医药ETF", "515030.SH": "新能源车ETF", "516160.SH": "新能源ETF",
    "110011.OF": "易方达优质精选", "161725.OF": "招商白酒指数", "000300.OF": "嘉实沪深300联接",
}

_FUTURE_PRODUCT: dict[str, str] = {
    "RB": "螺纹钢", "HC": "热轧卷板", "I": "铁矿石", "J": "焦炭", "JM": "焦煤",
    "CU": "铜", "AL": "铝", "ZN": "锌", "NI": "镍", "AU": "黄金", "AG": "白银",
    "SC": "原油", "FU": "燃料油", "BU": "沥青", "RU": "天然橡胶", "MA": "甲醇",
    "TA": "PTA", "SR": "白糖", "CF": "棉花", "SP": "纸浆", "EG": "乙二醇",
    "IF": "沪深300股指", "IC": "中证500股指", "IM": "中证1000股指", "IH": "上证50股指",
}

# 交易所后缀 -> 市场标识（供 AKShare 期货现货查询使用）
_FUTURE_EXCH_MAP: dict[str, str] = {
    "SHF": "SH", "CFE": "CFFEX", "DCE": "DL", "CZC": "ZJ", "INE": "SH", "GFE": "GF",
}


def _pinyin_initials(name: str) -> str:
    """取中文名每个字的拼音首字母，小写。无 pypinyin 时返回空字符串。"""
    if not name or not _HAS_PYPINYIN:
        return ""
    try:
        return "".join(p[0].lower() for p in lazy_pinyin(name) if p)
    except Exception:
        return ""


def _future_samples() -> list[tuple[str, str]]:
    """为内置期货品种生成当前年月附近的示例合约（供离线搜索）。"""
    now = datetime.now()
    # 常见主力月份映射（简化）
    month_map: dict[str, list[int]] = {
        "RB": [1, 5, 10], "HC": [1, 5, 10], "I": [1, 5, 9], "J": [1, 5, 9],
        "JM": [1, 5, 9], "CU": [1, 3, 5, 7, 9, 11], "AL": [1, 3, 5, 7, 9, 11],
        "ZN": [1, 3, 5, 7, 9, 11], "NI": [1, 3, 5, 7, 9, 11],
        "AU": [6, 8, 10, 12], "AG": [6, 8, 10, 12], "SC": [1, 3, 5, 7, 9, 11],
        "FU": [1, 3, 5, 7, 9, 11], "BU": [6, 9, 12], "RU": [1, 5, 9],
        "MA": [1, 5, 9], "TA": [1, 5, 9], "SR": [1, 5, 9], "CF": [1, 5, 9],
        "SP": [1, 3, 5, 7, 9, 11], "EG": [1, 5, 9],
        "IF": [3, 6, 9, 12], "IC": [3, 6, 9, 12], "IM": [3, 6, 9, 12], "IH": [3, 6, 9, 12],
    }
    samples: list[tuple[str, str]] = []
    for product, name in _FUTURE_PRODUCT.items():
        months = month_map.get(product, [1, 5, 9])
        # 取未来最近月份
        candidates = []
        for offset in (0, 1, 2):
            yy = (now.year + offset) % 100
            for m in months:
                candidates.append(f"{product}{yy:02d}{m:02d}")
        # 只取 2 个示例
        for i, suffix in enumerate(candidates[:2]):
            exch = "SHF" if product in {"RB", "HC", "I", "J", "JM", "CU", "AL", "ZN", "NI", "AU", "AG", "SC", "FU", "BU", "RU", "SP", "EG"} else (
                "CFE" if product in {"IF", "IC", "IM", "IH"} else (
                    "DCE" if product in {"J", "JM", "I"} else (
                        "CZC" if product in {"MA", "TA", "SR", "CF"} else "SHF"
                    )
                )
            )
            sym = f"{suffix}.{exch}"
            samples.append((sym, f"{name}{suffix[-4:]}"))
    return samples


def _option_samples() -> list[tuple[str, str]]:
    """为部分 ETF/指数生成示例期权合约（离线搜索用）。"""
    underlyings = [
        ("510300.SH", "沪深300ETF"), ("510050.SH", "上证50ETF"),
        ("159915.SZ", "创业板ETF"), ("510500.SH", "中证500ETF"),
    ]
    samples: list[tuple[str, str]] = []
    for base, name in underlyings:
        # 示例平值认购/认沽
        for otype, label in (("C", "购"), ("P", "沽")):
            sym = f"{base.replace('.SH', '').replace('.SZ', '')}{otype}4000.SH"
            samples.append((sym, f"{name} {label}4000"))
    return samples


def _build_search_index() -> list[dict]:
    """构建离线可搜索索引（symbol / 中文名 / 拼音首字母）。"""
    idx: list[dict] = []
    for sym, name in _STOCK_NAMES.items():
        idx.append({
            "symbol": sym, "name": name,
            "initials": _pinyin_initials(name),
            "asset": "stock",
            "tags": ["拼音首字母"],
        })
    for sym, name in _FUND_NAMES.items():
        idx.append({
            "symbol": sym, "name": name,
            "initials": _pinyin_initials(name),
            "asset": "fund",
            "tags": ["基金"],
        })
    for sym, name in _future_samples():
        idx.append({
            "symbol": sym, "name": name,
            "initials": _pinyin_initials(name),
            "asset": "future",
            "tags": ["期货"],
        })
    for sym, name in _option_samples():
        idx.append({
            "symbol": sym, "name": name,
            "initials": _pinyin_initials(name),
            "asset": "option",
            "tags": ["期权"],
        })
    return idx


# 全局只构建一次
_SEARCH_INDEX: list[dict] = _build_search_index()

# 在线缓存数据库（与数据缓存共用，减少文件碎片化）
_CACHE_DB = Path(__file__).resolve().parent.parent.parent / "data_cache.db"
_CACHE_TTL_SECONDS = 24 * 3600  # 1 天


def _ensure_search_table(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS symbol_search_cache (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            initials TEXT,
            asset TEXT NOT NULL,
            tags TEXT,
            updated_at REAL NOT NULL
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_search_initials ON symbol_search_cache(initials)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_search_name ON symbol_search_cache(name)")
    db.commit()


def _load_cached_index(db: sqlite3.Connection) -> list[dict]:
    now = datetime.now().timestamp()
    cur = db.execute(
        "SELECT symbol, name, initials, asset, tags FROM symbol_search_cache WHERE updated_at > ?",
        (now - _CACHE_TTL_SECONDS,),
    )
    rows = []
    for sym, name, initials, asset, tags in cur.fetchall():
        rows.append({
            "symbol": sym, "name": name, "initials": initials or _pinyin_initials(name),
            "asset": asset, "tags": (tags or "").split(",") if tags else [],
        })
    return rows


def _save_cached_index(db: sqlite3.Connection, rows: list[dict]) -> None:
    now = datetime.now().timestamp()
    db.execute("DELETE FROM symbol_search_cache")
    for r in rows:
        db.execute(
            "INSERT OR REPLACE INTO symbol_search_cache (symbol, name, initials, asset, tags, updated_at) VALUES (?,?,?,?,?,?)",
            (r["symbol"], r["name"], r.get("initials", ""), r["asset"], ",".join(r.get("tags", [])), now),
        )
    db.commit()


def _guess_board(symbol: str) -> list[str]:
    """根据代码推断板块标签（创业板 / 科创板 / 北交所）。"""
    code = _bare_code(symbol)
    tags = ["拼音首字母"]
    if code.startswith("300") or code.startswith("301"):
        tags.append("创业板")
    elif code.startswith("688") or code.startswith("689"):
        tags.append("科创板")
    elif code.startswith("8") or code.startswith("4"):
        tags.append("北交所")
    elif code.startswith("6"):
        tags.append("沪A")
    elif code.startswith("0") or code.startswith("3"):
        tags.append("深A")
    return tags


def _network_reachable(timeout: float = 2.0) -> bool:
    """快速探测外网是否可达，避免 akshare 在断网时长时间重试。"""
    try:
        import socket
        with socket.create_connection(("www.baidu.com", 443), timeout=timeout):
            return True
    except Exception:
        return False


def _fetch_akshare_index() -> list[dict] | None:
    """从 AKShare 拉取全市场 A 股 + 场内基金名称，生成可搜索索引；失败返回 None。"""
    if not _network_reachable():
        return None
    try:
        import akshare as ak
        rows: list[dict] = []
        # A 股
        df = ak.stock_info_a_code_name()
        if df is not None and not df.empty and {"代码", "名称"}.issubset(df.columns):
            for _, r in df.iterrows():
                code = str(r["代码"]).strip()
                name = str(r["名称"]).strip()
                if not code or not name:
                    continue
                exch = "SH" if code.startswith("6") or code.startswith("5") or code.startswith("688") else "SZ"
                sym = f"{code}.{exch}"
                rows.append({
                    "symbol": sym, "name": name,
                    "initials": _pinyin_initials(name),
                    "asset": "stock",
                    "tags": _guess_board(sym),
                })
        # 场内 ETF/LOF
        try:
            df_etf = ak.fund_etf_spot_em()
            if df_etf is not None and not df_etf.empty and {"代码", "名称"}.issubset(df_etf.columns):
                for _, r in df_etf.iterrows():
                    code = str(r["代码"]).strip()
                    name = str(r["名称"]).strip()
                    if not code or not name:
                        continue
                    exch = "SH" if code.startswith("5") else "SZ"
                    rows.append({
                        "symbol": f"{code}.{exch}", "name": name,
                        "initials": _pinyin_initials(name),
                        "asset": "fund",
                        "tags": ["ETF"],
                    })
        except Exception:
            pass
        return rows if rows else None
    except Exception:
        return None


def _get_live_index() -> list[dict]:
    """优先返回缓存/在线搜索索引，失败回退到离线内置索引。"""
    try:
        db = sqlite3.connect(str(_CACHE_DB), timeout=5.0)
        _ensure_search_table(db)
        cached = _load_cached_index(db)
        # 无论走缓存还是在线，都把内置映射合并进来（覆盖常见品种 + 兜底）
        cached_syms = set()
        if cached:
            cached_syms = {r["symbol"] for r in cached}
            for r in _SEARCH_INDEX:
                if r["symbol"] not in cached_syms:
                    cached.append(r)
            return cached
        live = _fetch_akshare_index()
        if live:
            live_syms = {r["symbol"] for r in live}
            for r in _SEARCH_INDEX:
                if r["symbol"] not in live_syms:
                    live.append(r)
            _save_cached_index(db, live)
            return live
        return _SEARCH_INDEX
    except Exception:
        return _SEARCH_INDEX


def search_symbols(query: str, asset: str | None = None, top_k: int = 10) -> list[dict]:
    """按代码 / 中文名 / 拼音首字母搜索内置标的。

    示例：``search_symbols('hmd')`` 可匹配"黑牡丹 / 恒铭达 / 宏明电子"。

    Parameters
    ----------
    query: 搜索关键字（支持代码、中文名子串、拼音首字母）。
    asset: 限制资产类别，None 表示全部。
    top_k: 最多返回条数。

    Returns
    -------
    list[dict]: 每个元素含 symbol / name / initials / asset / tags。
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    index = _get_live_index()
    results: list[tuple[int, dict]] = []
    for item in index:
        if asset and item["asset"] != asset:
            continue
        sym = item["symbol"].lower()
        name = item["name"].lower()
        initials = item["initials"].lower()
        score = 0
        if q == sym:
            score = 100
        elif q == name:
            score = 95
        elif initials and q == initials:
            score = 90
        elif sym.startswith(q):
            score = 80
        elif initials and initials.startswith(q):
            score = 70
        elif q in name:
            score = 60
        elif q in sym:
            score = 50
        if score:
            results.append((-score, item))
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results[:top_k]]


def _norm(symbol: str) -> str:
    return symbol.strip().upper()


def _bare_code(symbol: str) -> str:
    """600524.SH -> 600524；RB2410.SHF -> RB2410。"""
    return symbol.split(".")[0].upper()


def _detect_asset(symbol: str) -> str:
    s = _norm(symbol)
    if s.endswith(".OF"):
        return "fund"
    parts = s.rsplit(".", 1)
    exch = parts[-1] if len(parts) > 1 else ""
    if exch in {"SHF", "CFE", "DCE", "CZC", "INE", "GFE"}:
        return "future"
    if re.match(r"^[A-Z0-9]+[CP]\d+\.[A-Z]+$", s):
        return "option"
    return "stock"


def _ak_stock_name(code: str) -> str | None:
    try:
        import akshare as ak
        df = ak.stock_individual_info_em(symbol=code)
        if df is not None and not df.empty:
            row = df.set_index("item")["value"]
            for key in ("股票简称", "股票名称", "名称"):
                if key in row.index:
                    return str(row[key]).strip()
    except Exception:
        pass
    return None


def _ak_fund_name(code: str) -> str | None:
    try:
        import akshare as ak
        # ETF / 场内基金
        df = ak.fund_etf_spot_em()
        if df is not None and not df.empty:
            hits = df[df["代码"] == code]
            if not hits.empty:
                return str(hits.iloc[0].get("名称", "")).strip()
        # 开放式基金
        df2 = ak.fund_open_fund_daily_em()
        if df2 is not None and not df2.empty:
            hits2 = df2[df2["基金代码"] == code]
            if not hits2.empty:
                return str(hits2.iloc[0].get("基金简称", "")).strip()
    except Exception:
        pass
    return None


def _ak_future_name(symbol: str) -> str | None:
    """期货名称查询（AKShare 接口较复杂，优先本地规则）。"""
    return None


def _future_name(symbol: str) -> str:
    s = _norm(symbol)
    code = _bare_code(s)
    m = re.match(r"^([A-Z]+)(\d+)$", code)
    if not m:
        return code
    product, month = m.group(1), m.group(2)
    # 主连/连续合约特殊处理
    if month == "0":
        return f"{_FUTURE_PRODUCT.get(product, product)}主连"
    return f"{_FUTURE_PRODUCT.get(product, product)}{month}"


def _option_name(symbol: str) -> str:
    s = _norm(symbol)
    m = re.match(r"^(?P<base>[A-Z0-9]+?)(?P<otype>[CP])(?P<strike>\d+)\.(?P<exch>[A-Z]+)$", s)
    if not m:
        return s
    base = m.group("base")
    otype = "购" if m.group("otype") == "C" else "沽"
    strike = int(m.group("strike"))
    if base.isdigit() and len(base) == 6:
        strike = strike / 1000.0
    return f"{base} {otype}{strike}"


def get_symbol_name(symbol: str, asset: str | None = None) -> str:
    """返回 symbol 的中文名称；查不到时返回规则化兜底名称。"""
    s = _norm(symbol)
    if not s:
        return ""
    if s in _CACHE:
        return _CACHE[s]

    asset = asset or _detect_asset(s)
    name: str | None = None

    if asset == "fund":
        name = _FUND_NAMES.get(s)
        if not name:
            code = _bare_code(s)
            name = _ak_fund_name(code)
        if not name:
            name = f"基金 {s}"
    elif asset == "future":
        name = _future_name(s)
    elif asset == "option":
        name = _option_name(s)
    else:
        # stock / index
        name = _STOCK_NAMES.get(s)
        if not name:
            code = _bare_code(s)
            # 指数代码规则兜底
            if code.startswith("000") or code.startswith("880"):
                name = f"上证指数 {code}"
            elif code.startswith("399"):
                name = f"深证指数 {code}"
            else:
                name = _ak_stock_name(code)
        if not name:
            name = f"股票 {s}"

    _CACHE[s] = name
    return name


def get_symbols_names(symbols: list[str], asset: str | None = None) -> dict[str, str]:
    """批量查询代码名称，返回 {symbol: name}。"""
    return {s: get_symbol_name(s, asset) for s in symbols}


def build_name_table(lines_text: str, default_asset: str | None = None) -> pd.DataFrame:
    """把多行 "代码,权重" 或 "代码" 文本解析成带名称的 DataFrame。"""
    rows = []
    for line in lines_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        sym = parts[0]
        weight = parts[1] if len(parts) > 1 else ""
        rows.append({"标的代码": sym, "标的名称": get_symbol_name(sym, default_asset), "权重": weight})
    return pd.DataFrame(rows)


__all__ = ["get_symbol_name", "get_symbols_names", "build_name_table", "search_symbols"]
