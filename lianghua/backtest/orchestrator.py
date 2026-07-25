"""统一编排器：一键对「多资产 + 多策略」组合做回测并汇总绩效。

把 runner.run_backtest 的能力扩展成"研究计划(plan)"级别：
- plan 是一组 (symbol, strategy, asset, weight, start, end)
- 逐个回测 -> 按权重合并为组合净值 -> 输出组合指标 + 各成分明细

这样在 UI/CLI 里即可"一键编排"一个多资产配置方案，而不必单个标的来回跑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..perf.metrics import sharpe, max_drawdown, annual_return, summary
from ..portfolio.registry import get_optimizer, OPTIMIZER_INFO
from .runner import run_backtest


def run_plan(
    plan: list[dict],
    init_cash: float = 1_000_000.0,
    gw=None,
    optimizer: str | None = None,
    optimizer_kw: dict | None = None,
    report: bool = False,
) -> dict:
    """一键编排多资产回测。

    参数
    ----
    plan : list[dict]
        每个元素形如:
        {
          "symbol": "600519.SH",
          "strategy": "sma_cross",     # 可选，默认 sma_cross（来自统一策略注册表）
          "asset": "stock",            # 可选，默认按代码识别
          "weight": 0.4,               # 可选，默认等权；指定 optimizer 时本字段被覆盖
          "start": "2023-01-01",       # 可选，默认使用组合最早 start
          "end": "2023-12-31",         # 可选，默认使用组合最晚 end
        }
    init_cash : 组合初始资金
    gw : 可选注入数据网关（离线验证用）
    optimizer : 可选，组合优化器名（来自 portfolio.registry，
                如 risk_parity / min_variance / hrp / equal_risk_contrib /
                rotation / dual_momentum ...）。指定后按各成分收益矩阵算权重。
    optimizer_kw : 传给优化器的额外参数（如 {"top": 2} / {"target_vol": 0.10}）。
    report : 是否额外产出 tearsheet 报告 dict。

    返回
    ----
    dict:
        "equity"      -> 组合净值 pd.Series
        "weights"     -> {symbol: 权重}
        "components"  -> {symbol: {"equity","metrics","asset","strategy"}}
        "metrics"     -> 组合绩效摘要 dict
        "optimizer"   -> 实际使用的优化器名（None=计划权重）
        "report"      -> tearsheet DataFrame（report=True 时）
        "plan"        -> 回显用的规范化 plan
    """
    if not plan:
        raise ValueError("plan 不能为空")

    # 规范化：补齐 weight / start / end / strategy / asset
    default_start = min(p.get("start", "2000-01-01") for p in plan)
    default_end = max(p.get("end", "2100-01-01") for p in plan)
    norm_plan = []
    for p in plan:
        norm_plan.append({
            "symbol": p["symbol"],
            "strategy": p.get("strategy", "sma_cross"),
            "asset": p.get("asset"),
            "weight": float(p.get("weight", 1.0)),
            "start": p.get("start", default_start),
            "end": p.get("end", default_end),
        })
    total_w = sum(x["weight"] for x in norm_plan) or 1.0

    components: dict[str, dict] = {}
    norm_equities: list[pd.Series] = []
    weights_used: list[float] = []

    for spec in norm_plan:
        res = run_backtest(
            symbol=spec["symbol"],
            start=spec["start"],
            end=spec["end"],
            strategy=spec["strategy"],
            asset=spec["asset"],
            init_cash=init_cash,
            gw=gw,
        )
        eq = res.equity
        # 归一化到 1.0 起点，便于按权重合并
        norm = eq / eq.iloc[0]
        components[spec["symbol"]] = {
            "equity": eq,
            "metrics": summary(eq, res.trades),
            "asset": (res.asset.value if hasattr(res, "asset") else spec["asset"]),
            "strategy": spec["strategy"],
        }
        norm_equities.append(norm)
        weights_used.append(spec["weight"] / total_w)

    # 对齐到统一日期轴（取并集，前向填充；首个有效日前用 1.0 填充）
    aligned = pd.concat(norm_equities, axis=1)
    aligned.columns = [spec["symbol"] for spec in norm_plan]
    aligned = aligned.ffill().fillna(1.0)

    # 组合权重：优先用优化器，否则用计划权重
    used_optimizer = None
    if optimizer:
        if optimizer not in OPTIMIZER_INFO:
            raise ValueError(f"未知优化器: {optimizer}，可选: {list(OPTIMIZER_INFO)}")
        rets = aligned.pct_change().fillna(0.0)
        opt_fn = get_optimizer(optimizer, **(optimizer_kw or {}))
        w_series = opt_fn(rets)
        w_series = w_series.reindex(aligned.columns).fillna(0.0)
        s = w_series.sum()
        if abs(s) < 1e-9:
            # 退化情况（如全零/全 NaN）回退等权，避免组合被压平
            w_series = pd.Series(1.0 / len(aligned.columns), index=aligned.columns)
        else:
            w_series = w_series / s
        weights_used = w_series.values
        used_optimizer = optimizer
    w = np.array(weights_used)
    portfolio_norm = aligned.mul(w, axis=1).sum(axis=1)
    portfolio_equity = init_cash * portfolio_norm

    # 组合绩效
    metrics = summary(portfolio_equity, [])
    metrics["sharpe"] = sharpe(portfolio_equity)
    metrics["max_drawdown"] = max_drawdown(portfolio_equity)
    metrics["annual_return"] = annual_return(portfolio_equity)

    out = {
        "equity": portfolio_equity,
        "weights": {spec["symbol"]: round(float(weights_used[i]), 4) for i, spec in enumerate(norm_plan)},
        "components": components,
        "metrics": metrics,
        "optimizer": used_optimizer,
        "plan": norm_plan,
    }
    if report:
        from ..report.tearsheet import tearsheet
        out["report"] = tearsheet(portfolio_equity)
    return out
