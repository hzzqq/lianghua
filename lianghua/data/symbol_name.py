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
from pathlib import Path

import pandas as pd

# 缓存：symbol -> name
_CACHE: dict[str, str] = {}

# ---------- 内置名称映射（离线兜底） ----------
_STOCK_NAMES: dict[str, str] = {
    # 白酒
    "600519.SH": "贵州茅台", "000858.SZ": "五粮液", "000568.SZ": "泸州老窖",
    "002304.SZ": "洋河股份", "600779.SH": "水井坊",
    # 银行
    "601398.SH": "工商银行", "601318.SH": "中国平安", "600036.SH": "招商银行",
    "601166.SH": "兴业银行", "600000.SH": "浦发银行", "601328.SH": "交通银行",
    "601988.SH": "中国银行", "002142.SZ": "宁波银行",
    # 新能源 / 科技
    "300750.SZ": "宁德时代", "002594.SZ": "比亚迪", "601012.SH": "隆基绿能",
    "600438.SH": "通威股份", "300014.SZ": "亿纬锂能", "002415.SZ": "海康威视",
    "000725.SZ": "京东方A", "002230.SZ": "科大讯飞", "688981.SH": "中芯国际",
    "000063.SZ": "中兴通讯", "600524.SH": "岭南控股",
    # 医药
    "600276.SH": "恒瑞医药", "300760.SZ": "迈瑞医疗", "000661.SZ": "长春高新",
    "600196.SH": "复星医药", "002821.SZ": "凯莱英",
    # 上证50 / 沪深300 龙头
    "600900.SH": "长江电力", "601888.SH": "中国中免", "601628.SH": "中国人寿",
    "600030.SH": "中信证券", "600837.SH": "海通证券", "000651.SZ": "格力电器",
    "002475.SZ": "立讯精密", "600585.SH": "海螺水泥", "000625.SZ": "长安汽车",
    "600999.SH": "招商证券", "000001.SZ": "平安银行",
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


__all__ = ["get_symbol_name", "get_symbols_names", "build_name_table"]
