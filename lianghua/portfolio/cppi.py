"""CPPI 固定比例投资组合保险：底仓保护 + 风险乘数。

v2 打磨（self-driving polish）：
- 输入守卫：asset_returns 必须为有限一维收益序列；floor_pct∈[0,1)、multiplier>0、
  init_cash>0、periods>=1、risk_free 有限，否则抛清晰 ValueError，杜绝静默 NaN/死亡螺旋。
- 隐性修复：原实现对 NaN/inf 收益静默传播（value→NaN 后永不恢复）、floor_pct>=1 或
  multiplier<=0 等非法参数无校验；现一律显式拦截。
- 新增能力 transaction_cost：再平衡摩擦成本（按换手额计提），让 CPPI 更接近真实交易。
- 新增能力 cppi_summary：可观测快照（终值/最大回撤/最低净值/平均风险敞口/触底次数）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..perf.metrics import max_drawdown


def _validate_cppi(asset_returns, floor_pct, multiplier, init_cash, risk_free, periods):
    """CPPI 输入守卫：把"静默错误"转为清晰异常。"""
    if not isinstance(asset_returns, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
        raise TypeError("asset_returns 必须是收益序列（pd.Series / array / list）")
    r = pd.Series(np.asarray(asset_returns, dtype=float).ravel())
    if r.empty:
        raise ValueError("asset_returns 不能为空")
    if not np.all(np.isfinite(r.values)):
        bad = int((~np.isfinite(r.values)).sum())
        raise ValueError(f"asset_returns 含 {bad} 个非有限值（NaN/inf），无法计算 CPPI")
    if not (0.0 <= floor_pct < 1.0):
        raise ValueError(f"floor_pct 必须落在 [0, 1) 区间，收到 {floor_pct}")
    if multiplier <= 0:
        raise ValueError(f"multiplier 必须为正数，收到 {multiplier}")
    if init_cash <= 0:
        raise ValueError(f"init_cash 必须为正数，收到 {init_cash}")
    if not np.isfinite(risk_free):
        raise ValueError(f"risk_free 必须有限，收到 {risk_free}")
    if not isinstance(periods, (int, np.integer)) or periods < 1:
        raise ValueError(f"periods 必须为正整数，收到 {periods}")
    return r


def cppi(asset_returns: pd.Series, floor_pct: float = 0.8,
         multiplier: float = 3.0, init_cash: float = 1_000_000.0,
         risk_free: float = 0.0, periods: int = 252,
         transaction_cost: float = 0.0) -> pd.DataFrame:
    """返回含 equity / cushion / risky_weight 的 DataFrame。

    transaction_cost: 再平衡摩擦成本率（按每期换手额计提），默认 0。
    当风险敞口在相邻期之间调整时，对该部分金额扣减成本，避免"无摩擦幻觉"。
    """
    r = _validate_cppi(asset_returns, floor_pct, multiplier, init_cash, risk_free, periods)

    if not (0.0 <= transaction_cost < 1.0):
        raise ValueError(f"transaction_cost 必须落在 [0, 1) 区间，收到 {transaction_cost}")

    n = len(r)
    floor_val = floor_pct * init_cash
    value = float(init_cash)
    prev_risky_w = 0.0
    rows = []
    breaches = 0
    for t in range(n):
        floor_val *= (1.0 + risk_free / periods)
        cushion = value - floor_val
        if cushion < 0:
            breaches += 1
        risky_w = max(0.0, min(1.0, multiplier * cushion / value)) if value > 0 else 0.0

        # 再平衡摩擦：仅对"新增/减少的风险敞口"部分计提成本
        if transaction_cost > 0.0 and t > 0:
            turnover = abs(risky_w - prev_risky_w)
            value -= turnover * value * transaction_cost

        rows.append({"equity": value, "cushion": cushion, "risky_weight": risky_w})
        prev_risky_w = risky_w
        ret = float(r.iloc[t])
        value = value * (risky_w * (1.0 + ret) + (1.0 - risky_w))

    df = pd.DataFrame(rows)
    # 数值稳定：账本保留 6 位，避免浮点漂移累积
    for col in ("equity", "cushion", "risky_weight"):
        df[col] = df[col].round(6)
    df.attrs["floor_breaches"] = breaches
    return df[["equity", "cushion", "risky_weight"]]


def cppi_summary(df: pd.DataFrame, init_cash: float = 1_000_000.0) -> dict:
    """CPPI 运行结果的可观测快照（新增能力）。

    返回终值、收益率、最大回撤、最低净值、平均风险敞口、触底（击穿 floor）次数。
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df 必须为非空的 CPPI 结果 DataFrame")
    eq = pd.to_numeric(df["equity"], errors="coerce")
    if not np.all(np.isfinite(eq.values)):
        raise ValueError("equity 列含非有限值，无法汇总")
    final = float(eq.iloc[-1])
    # perf.metrics.max_drawdown 返回负数（回撤幅度），汇总里以正数幅度展示更直观
    mdd = abs(float(max_drawdown(eq)))
    min_eq = float(eq.min())
    avg_risky = float(pd.to_numeric(df["risky_weight"], errors="coerce").mean())
    breaches = int(df.attrs.get("floor_breaches", 0))
    return {
        "init_cash": float(init_cash),
        "final_equity": round(final, 2),
        "total_return": round((final - init_cash) / init_cash, 6),
        "max_drawdown": round(mdd, 6),
        "min_equity": round(min_eq, 2),
        "avg_risky_weight": round(avg_risky, 6),
        "floor_breaches": breaches,
    }
