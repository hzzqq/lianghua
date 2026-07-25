"""AI 因子模块：因子计算、评估与自动搜索（RD-Agent 风格，纯本地可跑）。"""
from __future__ import annotations

from .engine import FactorEngine
from .auto_search import FactorSearcher
from .returns import factor_long_short, factor_group_returns, factor_ic

__all__ = ["FactorEngine", "FactorSearcher",
           "factor_long_short", "factor_group_returns", "factor_ic"]
