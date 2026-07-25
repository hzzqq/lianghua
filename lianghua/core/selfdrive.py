"""自驱动循环接入层：把 capability 注册表变成「下一步该扩展什么」的生成器。

self-driving-dev 等自主循环 skill 调用 ``next_gaps(n)`` 即可拿到一批
「尚未实现、但符合平台扩展规范」的能力缺口清单（category/name/kind/rationale），
逐项实现并登记进对应 registry / capabilities 后即可被 UI、CLI、编排器自动发现。

设计原则：
- 不硬编码能力列表在多处；策略与优化器动态取自注册表，其余取自 capabilities 索引。
- 候选池是「yet-to-build」清单，实现一个就从池里划掉一个（通过差集计算）。
- 纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

from .capabilities import list_capabilities, summary_counts

# 各能力类别的目标规模（用于决定要不要继续往该类扩）
_TARGETS = {
    "strategies": 40,
    "optimizers": 28,
    "indicators": 24,
    "perf_functions": 52,
    "risk_functions": 40,
    "factor_functions": 20,
}

# 「尚未实现」的候选池（差集计算可避免重复）。命名与现有模块风格保持一致。
_STRATEGY_POOL = [
    "parabolic_sar", "adx_trend", "cci_signal", "roc", "ultimate_oscillator",
    "pivot_points", "heikin_ashi", "renko_trend", "demark", "klinger",
    "mass_index", "chaikin_volatility", "vwap_trend", "boll_keltner",
    "chande_momentum", "force_index", "know_sure_thing", "mcclellan",
]
_OPTIMIZER_POOL = [
    "min_tail_risk", "vol_target_opt", "bayesian_shrinkage", "max_return",
    "min_track_error", "robust_cov", "hierarchical_cluster_rp", "equal_risk_ewm",
    "risk_parity_clustered", "max_diversification_ewm",
]
_INDICATOR_POOL = [
    "adx", "pivot", "heikin_ashi", "renko", "demarker", "klinger",
    "mass_index", "chaikin_vol", "force_index", "know_sure_thing",
    "standard_pivot", "zigzag", "donchian_channel", "price_channels",
]
_PERF_POOL = [
    "omega_ratio", "calmar_ratio", "tail_ratio", "payoff_ratio",
    "common_sense_ratio", "outlier_ratio", "hit_ratio", "profit_factor",
    "recovery_factor", "sterling_ratio", "burke_ratio", "sharpe_penalized",
]
_RISK_POOL = [
    "tail_dependence", "risk_contribution", "entropy_risk", "concentration_risk",
    "correlation_risk", "marginal_var", "incremental_var", "expected_shortfall",
    "liquidity_adjusted_var", "regime_var", "stress_var", "max_loss_prob",
]
_FACTOR_POOL = [
    "factor_neutrality", "factor_turnover", "factor_icir", "factor_decay",
    "factor_corr", "factor_orthogonalize", "factor_weight_decay",
    "factor_cross_section", "factor_portfolio", "factor_decay_halflife",
]


def _implemented() -> dict:
    cap = list_capabilities()
    return {
        "strategies": set(cap["strategies"].keys()),
        "optimizers": set(cap["optimizers"].keys()),
        "indicators": set(cap["indicators"]),
        "perf_functions": {f for v in cap["perf"].values() for f in v},
        "risk_functions": {f for v in cap["risk"].values() for f in v},
        "factor_functions": {f for v in cap["factor"].values() for f in v},
    }


def deficit() -> dict:
    """返回每类『还差多少到目标』。"""
    counts = summary_counts()
    impl = _implemented()
    out = {}
    for cat, target in _TARGETS.items():
        have = counts.get(cat, 0)
        out[cat] = {
            "have": have,
            "target": target,
            "gap_to_target": max(0, target - have),
            "pending": sorted(set({
                "strategies": _STRATEGY_POOL,
                "optimizers": _OPTIMIZER_POOL,
                "indicators": _INDICATOR_POOL,
                "perf_functions": _PERF_POOL,
                "risk_functions": _RISK_POOL,
                "factor_functions": _FACTOR_POOL,
            }[cat]) - impl[cat]),
        }
    return out


def next_gaps(n: int = 10) -> list[dict]:
    """生成下一批可扩展能力缺口（最多 n 个），优先补离目标最近的类。

    返回项形如：
        {"category": "strategies", "name": "parabolic_sar",
         "kind": "df", "rationale": "..."}
    便于 self-driving-dev 逐项实现并登记。
    """
    d = deficit()
    # 按 (gap_to_target 降序, pending 数量降序) 排，先补最缺的类
    order = sorted(
        d.items(),
        key=lambda kv: (kv[1]["gap_to_target"], len(kv[1]["pending"])),
        reverse=True,
    )
    gaps: list[dict] = []
    by_cat = {
        "strategies": ("df", "函数式择时/趋势/反转策略，输出 {-1,0,1}"),
        "optimizers": ("returns", "吃收益矩阵产出和为1的多头权重 Series"),
        "indicators": ("df", "输入 OHLCV 输出等长 Series/DataFrame"),
        "perf_functions": ("equity", "输入净值或收益，输出标量风险/绩效指标"),
        "risk_functions": ("returns", "输入收益/权重，输出风险分解或尾部指标"),
        "factor_functions": ("factor", "输入因子面板，输出中性化/换手/IC 类指标"),
    }
    for cat, info in order:
        kind, why = by_cat[cat]
        for name in info["pending"]:
            if len(gaps) >= n:
                return gaps
            gaps.append({
                "category": cat,
                "name": name,
                "kind": kind,
                "rationale": f"{cat[:-1]} 类缺口：{why}（当前 {info['have']}/{info['target']}）",
            })
    return gaps


def report() -> str:
    """人类可读的缺口报告。"""
    d = deficit()
    lines = ["[自驱动能力缺口]"]
    for cat, info in d.items():
        lines.append(
            f"- {cat}: {info['have']}/{info['target']} "
            f"(还差 {info['gap_to_target']})，待办 {len(info['pending'])} 项"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
    print("\n--- next 10 gaps ---")
    for g in next_gaps(10):
        print(g)
