"""止盈：固定目标 / 移动分段。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def take_profit_levels(entry: float, levels: list, ratios: list) -> dict:
    """按 entry + 多个盈利目标比例返回 止盈价 → 减仓比例。

    levels/ratios 必须等长；ratios 为各档减仓比例（建议合计≈1）。
    """
    if len(levels) != len(ratios):
        raise ValueError(f"levels({len(levels)}) 与 ratios({len(ratios)}) 长度不一致")
    return {entry * (1 + lv): rt for lv, rt in zip(levels, ratios)}


def scaled_exit(price: pd.Series, entry: float, levels: list,
                ratios: list | None = None) -> pd.Series:
    """价格逐档触及盈利目标后，累计已减仓比例（0→1）。

    每档仅在首次触及时按对应 ratios 减仓一次并锁定，累计不超过 1。
    ratios 缺省为各档均分、末档补足到 1（全平）。

    与 take_profit_levels 共用同一套 levels/ratios，修复了旧版忽略 ratios、
    用 tgt*10 硬编码导致单档即全平的缺陷。
    """
    levels = list(levels)
    if ratios is None:
        if not levels:
            return pd.Series(0.0, index=price.index, name="scaled_exit")
        base = 1.0 / len(levels)
        ratios = [base] * (len(levels) - 1) + [1.0 - base * (len(levels) - 1)]
    else:
        ratios = list(ratios)
        if len(levels) != len(ratios):
            raise ValueError(f"levels({len(levels)}) 与 ratios({len(ratios)}) 长度不一致")
    order = sorted(range(len(levels)), key=lambda i: levels[i])
    cum = 0.0
    done = [False] * len(levels)
    out = []
    for p in price:
        for i in order:
            if not done[i] and p >= entry * (1 + levels[i]):
                cum += ratios[i]
                done[i] = True
        out.append(min(1.0, cum))
    return pd.Series(out, index=price.index, name="scaled_exit")

