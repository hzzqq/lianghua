"""自动因子搜索：遍历内置因子库，计算每个因子的预测力(IC / 分层多空)，按 |IC| 排序输出 Top N。

RD-Agent 风格：自动生成候选 -> 量化打分 -> 排序筛选。纯本地实现，无需外部 LLM。
（若要接 LLM 自动写因子表达式，可在 search() 中扩展候选生成器。）
"""
from __future__ import annotations

import pandas as pd

from .engine import FactorEngine


class FactorSearcher:
    def __init__(self, df: pd.DataFrame, engine: FactorEngine | None = None):
        self.engine = engine or FactorEngine(df)
        self.df = df

    def search(self, top_n: int = 5, forward: int = 5, min_ic: float = 0.02) -> list[dict]:
        """自动评估所有候选因子，过滤弱因子，按 |IC| 降序返回 Top N。"""
        results = []
        for name in FactorEngine.FACTORS:
            try:
                f = self.engine.compute(name)
                r = self.engine.evaluate(f, forward)
                r["factor"] = name
                results.append(r)
            except Exception:
                continue
        filtered = [r for r in results if pd.notna(r["ic"]) and abs(r["ic"]) >= min_ic]
        filtered.sort(key=lambda x: abs(x["ic"]), reverse=True)
        if filtered:
            return filtered[:top_n]
        # 全部低于阈值时仍返回排名靠前的弱因子，便于诊断（真实数据通常有显著因子）
        results.sort(key=lambda x: abs(x["ic"]) if pd.notna(x["ic"]) else -1, reverse=True)
        return results[:top_n]

    def top_factor_backtest(self, top_n: int = 3, forward: int = 5) -> dict[str, pd.Series]:
        """返回 Top 因子的单因子回测净值序列，便于可视化对比。"""
        out: dict[str, pd.Series] = {}
        for r in self.search(top_n=top_n, forward=forward):
            f = self.engine.compute(r["factor"])
            out[r["factor"]] = self.engine.backtest_factor(f)
        return out
