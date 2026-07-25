"""压力测试：对持仓施加历史/假设情景冲击，估算组合损益。

- stress_test      : 逐情景计算组合损益%(%) 与可选绝对金额。
- stress_report    : 汇总表 + 最差情景（新增可观测能力）。
- BUILTIN_SCENARIOS: 内置历史情景库（2008/COVID/加息/衰退），可直接按名引用。

隐性修复：原实现对非有限权重/冲击静默产出 NaN 损益，且无绝对值口径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["stress_test", "stress_report", "list_builtin_scenarios", "BUILTIN_SCENARIOS"]

# 内置情景库：{资产: 冲击比例}（示例性历史近似，供快速压力测试）
BUILTIN_SCENARIOS: dict[str, dict] = {
    "2008_GFC": {"股票": -0.45, "债券": 0.05, "商品": -0.30, "房地产": -0.40, "现金": 0.0},
    "COVID_2020": {"股票": -0.34, "债券": 0.08, "商品": -0.40, "房地产": -0.25, "现金": 0.0},
    "加息冲击": {"股票": -0.15, "债券": -0.10, "商品": 0.05, "房地产": -0.12, "现金": 0.01},
    "衰退": {"股票": -0.25, "债券": 0.10, "商品": -0.15, "房地产": -0.20, "现金": 0.0},
}


def list_builtin_scenarios() -> list:
    """返回可用内置情景名列表（新增能力）。"""
    return list(BUILTIN_SCENARIOS.keys())


def _resolve_scenario(shock) -> dict:
    if isinstance(shock, str):
        if shock not in BUILTIN_SCENARIOS:
            raise KeyError(f"未知内置情景: {shock}")
        return BUILTIN_SCENARIOS[shock]
    if not isinstance(shock, dict):
        raise TypeError("情景必须是 {资产: 冲击} 字典或内置情景名字符串")
    return shock


def _validate_weights(weights: dict) -> dict:
    if not isinstance(weights, dict):
        raise TypeError("weights 必须是 {资产: 权重} 字典")
    for a, w in weights.items():
        if not np.isfinite(float(w)):
            raise ValueError(f"权重 {a}={w} 非有限")
    return weights


def stress_test(weights: dict, scenarios: dict, base_value: float | None = None) -> pd.DataFrame:
    """weights: {资产: 权重}；scenarios: {情景名: {资产: 冲击}} 或 {情景名: 内置名}。

    返回 DataFrame：情景名 + 组合损益(%)；若给 base_value 则附绝对金额列。
    """
    w = _validate_weights(weights)
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("scenarios 必须是非空字典")
    if base_value is not None:
        bv = float(base_value)
        if not np.isfinite(bv) or bv <= 0:
            raise ValueError("base_value 必须为正有限值")
    rows = []
    for name, raw in scenarios.items():
        shock = _resolve_scenario(raw)
        for a, s in shock.items():
            if not np.isfinite(float(s)):
                raise ValueError(f"情景 {name} 中 {a} 冲击非有限")
        pnl = sum(w.get(a, 0.0) * shock.get(a, 0.0)
                  for a in set(w) | set(shock))
        row = {"情景": name, "组合损益%": pnl * 100}
        if base_value is not None:
            row["组合损益金额"] = pnl * bv
        rows.append(row)
    return pd.DataFrame(rows)


def stress_report(weights: dict, scenarios: dict,
                  base_value: float | None = None) -> dict:
    """返回 {table, worst_scenario, worst_pnl_pct}（新增可观测能力）。"""
    df = stress_test(weights, scenarios, base_value=base_value)
    if df.empty:
        return {"table": df, "worst_scenario": None, "worst_pnl_pct": 0.0}
    worst = df.loc[df["组合损益%"].idxmin()]
    return {
        "table": df,
        "worst_scenario": worst["情景"],
        "worst_pnl_pct": float(worst["组合损益%"]),
    }
