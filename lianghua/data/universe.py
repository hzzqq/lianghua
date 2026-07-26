"""成分股 / 行业 universe（迭代 181+，超自驱动目标）。

内置若干常见指数与行业成分示例（离线可用，作为演示 universe）。
真实成分可通过 AKShare/BaoStock 动态拉取时再覆盖。
"""
from __future__ import annotations

UNIVERSES: dict[str, list[str]] = {
    "沪深300": ["600519.SH", "601318.SH", "600036.SH", "000858.SZ", "601166.SH",
               "600276.SH", "000333.SZ", "600900.SH", "601888.SH", "002415.SZ"],
    "上证50": ["600519.SH", "601318.SH", "600036.SH", "601166.SH", "600276.SH",
              "600900.SH", "601398.SH", "601628.SH", "600030.SH", "601888.SH"],
    "中证500": ["000001.SZ", "002142.SZ", "002594.SZ", "600837.SH", "000651.SZ",
               "002475.SZ", "600585.SH", "000625.SZ", "002230.SZ", "600999.SH"],
    "白酒": ["600519.SH", "000858.SZ", "000568.SZ", "002304.SZ", "600779.SH"],
    "银行": ["601398.SH", "601318.SH", "600036.SH", "601166.SH", "600000.SH",
            "601328.SH", "601988.SH", "002142.SZ"],
    "新能源": ["300750.SZ", "002594.SZ", "601012.SH", "600438.SH", "300014.SZ"],
    "科技": ["002415.SZ", "000725.SZ", "002230.SZ", "688981.SH", "000063.SZ"],
    "医药": ["600276.SH", "300760.SZ", "000661.SZ", "600196.SH", "002821.SZ"],
}


def list_universes() -> list[str]:
    """可用 universe 名称列表。"""
    return list(UNIVERSES.keys())


def get_universe(name: str) -> list[str]:
    """返回指定 universe 的成分代码列表。"""
    if name not in UNIVERSES:
        raise ValueError(f"未知 universe: {name}，可选 {list_universes()}")
    return list(UNIVERSES[name])


def in_universe(name: str, symbol: str) -> bool:
    """判定 symbol 是否属于指定 universe（成分股筛选原子能力）。"""
    if name not in UNIVERSES:
        raise ValueError(f"未知 universe: {name}，可选 {list_universes()}")
    return symbol in UNIVERSES[name]


def union(names: list[str] | None = None) -> list[str]:
    """多个 universe 的去重并集；不传则返回全部 universe 的去重并集。

    用于构建跨板块/宽基股票池，去重保证同一标的不会被重复纳入。
    """
    keys = names if names is not None else list(UNIVERSES.keys())
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k not in UNIVERSES:
            raise ValueError(f"未知 universe: {k}，可选 {list_universes()}")
        for s in UNIVERSES[k]:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


__all__ = ["UNIVERSES", "list_universes", "get_universe", "in_universe", "union"]
