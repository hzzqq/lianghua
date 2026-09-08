# -*- coding: utf-8 -*-
"""腾讯行情源（零新增依赖，仅标准库 urllib）。

腾讯公开行情接口在多数网络环境（含受限沙箱、AKShare 东财接口被墙、
BaoStock 超时的机器）均可达，且稳定覆盖 A股/ETF/指数日 K（前/后复权）与
实时快照。把它作为 AKShare / BaoStock 不可用时的兜底**真实源**，彻底解决
「真实行情取不到 → 静默降级演示数据 → 回测/信号建立在随机游走上、结论无意义」
这个让框架「实际使用效果差」的根因。

字段顺序（腾讯 fqkline 返回）：[date, open, close, high, low, volume]。
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd

_FQ_URL = (
    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get"
    "?param={code},{period},{start},{end},{limit},{fq}"
)
_QT_URL = "https://qt.gtimg.cn/q={code}"

_UA = {"User-Agent": "Mozilla/5.0 (compatible; lianghua/1.6)"}

# 复权类型 -> 腾讯返回的键名（缺省退化到 day）
_FQ_MAP = {"qfq": "qfqday", "hfq": "hfqday", "": "day"}


def _opener() -> urllib.request.OpenerDirector:
    """构造带代理（若环境提供）的 opener。

    受限网络（沙箱）下进程通过 ``https_proxy`` 环境变量走代理出网；urllib 的代理
    自动探测在「守护线程内取数」时会失效，导致直连被防火墙 DROP、表现为取数超时。
    因此显式从环境变量读取代理注入 ``ProxyHandler``。

    坑：只给 ``https`` 注册代理、不要给 ``http`` 也注册同名代理。本沙箱的代理仅支持
    CONNECT 隧道（https 走隧道），若同时把 https URL 注册成「普通代理转发」会得到
    ``HTTP 501 Not Implemented``。urllib 对 https+代理默认就是走 CONNECT 隧道，所以
    只挂 ``https`` 一项即可让 https 请求正确建立隧道（http 明文请求本就直达，无需代理）。
    """
    px = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    if px:
        return urllib.request.build_opener(urllib.request.ProxyHandler({"https": px}))
    return urllib.request.build_opener()


def _to_tencent_code(symbol: str) -> str | None:
    """代码映射：600519.SH -> sh600519；000001.SZ -> sz000001；510050.SH -> sh510050。

    腾讯仅提供场内品种（股票 / ETF / 指数 / 可转债）行情，场外开放式基金
    （``.OF``）无公开行情，返回 None 由上层网关降级到其它源或演示数据。
    """
    if "." not in symbol:
        return symbol
    code, ext = symbol.split(".", 1)
    ext = ext.lower()
    if ext in ("sh", "sz"):
        return ext + code
    # 其他后缀（of 等场外基金）：腾讯无行情
    return None


def fetch_daily(symbol: str, start: str, end: str, asset: str = "stock",
                fq: str = "qfq", limit: int = 320,
                timeout: float = 10.0, max_pages: int = 40) -> pd.DataFrame | None:
    """抓取日 K（前/后复权）。返回 OHLCV DataFrame，失败返回 None。

    腾讯单次最多返回约 ``limit`` 根 K 线，超出需分页：用本页最后一根日期作为
    下一页起点循环翻页，直到覆盖 ``end`` 或接口无更多数据。
    """
    tc = _to_tencent_code(symbol)
    if tc is None:
        return None
    rows: list = []
    cur_end = end
    for _ in range(max_pages):
        url = _FQ_URL.format(code=tc, period="day", start=start,
                             end=cur_end, limit=limit, fq=fq)
        try:
            req = urllib.request.Request(url, headers=_UA)
            with _opener().open(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            # 网络异常：已累计的页先返回，避免整段丢失
            break
        node = (payload or {}).get("data", {}).get(tc)
        if not node:
            break
        block = node.get(_FQ_MAP.get(fq, "day")) or node.get("day")
        if not block:
            break
        rows.extend(block)
        # 腾讯 fqkline 的 start 参数对长区间被忽略，只按 end 往前取 limit 根；
        # 因此翻页必须把 end 前移到本页第一根日期的前一天，而非后移 start。
        first_date = str(block[0][0])
        if first_date <= start:
            break
        cur_end = (pd.Timestamp(first_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if not rows:
        return None
    # 按日期升序、去重（翻页边界可能重叠），并截断到请求起点
    seen = set()
    out = []
    for r in sorted(rows, key=lambda x: x[0]):
        if r[0] in seen:
            continue
        seen.add(r[0])
        if r[0] >= start:
            out.append(r)
    return _build(out)


def _build(rows: list) -> pd.DataFrame | None:
    """把腾讯原始行规范为 OHLCV。

    腾讯 fqkline 行并非严格 6 列：部分带前缀的行情会返回 7 列
    （首列为空/序号），因此只取**前 6 列**并按
    [date, open, close, high, low, volume] 对齐，避免多余列导致构造报错。
    """
    clean = []
    for r in rows:
        if r is None:
            continue
        if len(r) >= 6:
            clean.append(list(r[:6]))
    if not clean:
        return None
    df = pd.DataFrame(clean, columns=["date", "open", "close", "high", "low", "volume"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0]
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def fetch_quote(symbol: str, timeout: float = 8.0) -> float | None:
    """实时快照：返回最新价 float，失败返回 None（解析 qt.gtimg.cn）。

    qt.gtimg.cn 返回 GBK 编码文本，形如
    ``v_sh600519="1~名称~代码~当前价~昨收~开盘~成交量~..."``，价格在第 4 个 ~ 字段。
    """
    tc = _to_tencent_code(symbol)
    if tc is None:
        return None
    url = _QT_URL.format(code=tc)
    try:
        req = urllib.request.Request(url, headers=_UA)
        with _opener().open(req, timeout=timeout) as resp:
            raw = resp.read().decode("gbk", errors="ignore")
    except Exception:
        return None
    if "=" not in raw:
        return None
    body = raw.split("=", 1)[1].strip().strip('"')
    parts = body.split("~")
    if len(parts) < 4:
        return None
    try:
        return float(parts[3])
    except (ValueError, TypeError):
        return None


__all__ = ["fetch_daily", "fetch_quote", "_to_tencent_code"]
