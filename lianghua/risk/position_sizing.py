"""仓位管理：凯利 / 固定分数 / 波动率目标（迭代打磨）。

纯 pandas/numpy。提供单笔下注规模测算，强调对非法输入的显式守卫，
避免产出负仓位 / inf 仓位等隐性陷阱。

迭代打磨（本轮回测收尾）：
- kelly_fraction 新增 fraction 参数（支持半凯利等分数凯利），并校验 win_rate ∈ [0,1]。
- fixed_fractional 守卫 equity/risk_pct/stop_dist 合法区间（负值/风险≥100% 返回 0）。
- vol_target_size 守卫 equity/price/daily_vol/target_vol 为正，杜绝负/inf 仓位。
- 新增 kelly_from_returns：从交易收益序列直接估计连续凯利分数 f = E[r]/Var[r]
  （数据驱动，机构更常用），支持分数凯利与有限值过滤。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["kelly_fraction", "fixed_fractional", "vol_target_size", "kelly_from_returns"]


def _to_float(x, name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是有限数字，收到 {x!r}")
    if not np.isfinite(v):
        raise ValueError(f"{name} 必须为有限数，收到 {x!r}")
    return v


def kelly_fraction(win_rate: float, win_odds: float, fraction: float = 1.0) -> float:
    """凯利分数 f = p - (1-p)/b，b=盈亏比（盈利/亏损）。返回 fraction * f（分数凯利）。

    fraction: 分数凯利系数（如 0.5 = 半凯利）；结果夹在 [0,1]。
    win_rate 越界(非 [0,1]) / fraction 越界 显式抛错；win_odds<=0 返回 0（不下注）。
    """
    p = _to_float(win_rate, "win_rate")
    b = _to_float(win_odds, "win_odds")
    fr = _to_float(fraction, "fraction")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"win_rate 必须在 [0,1]，收到 {win_rate!r}")
    if not (0.0 < fr <= 1.0):
        raise ValueError(f"fraction 必须介于 (0,1]，收到 {fraction!r}")
    if b <= 0:
        return 0.0
    f = p - (1.0 - p) / b
    return float(max(0.0, min(1.0, f * fr)))


def fixed_fractional(equity: float, risk_pct: float, stop_dist: float) -> float:
    """每笔风险 = equity*risk_pct，仓位 = 风险 / 止损距离。

    隐性修复：原对 equity<=0 / risk_pct 越界 / 负 stop_dist 未守卫，会产出
    负仓位或除零 inf；现统一返回 0（不下注）。
    """
    try:
        eq = float(equity)
        rp = float(risk_pct)
        sd = float(stop_dist)
    except (TypeError, ValueError):
        return 0.0
    if eq <= 0 or not (0.0 < rp < 1.0) or sd <= 0:
        return 0.0
    return float(eq * rp / sd)


def vol_target_size(equity: float, price: float, daily_vol: float,
                    target_vol: float = 0.01, periods: int = 252) -> float:
    """使单标的日波动≈target 的股数。periods 保留作年化处理扩展位（当前按日口径）。

    隐性修复：原 equity<=0 / target_vol<=0 / daily_vol<=0 未守卫，
    会产出负或 inf 仓位；现返回 0。
    """
    try:
        eq = float(equity)
        px = float(price)
        dv = float(daily_vol)
        tv = float(target_vol)
    except (TypeError, ValueError):
        return 0.0
    if eq <= 0 or px <= 0 or dv <= 0 or tv <= 0:
        return 0.0
    unit_vol = px * dv
    if unit_vol <= 0:
        return 0.0
    return float(tv * eq / unit_vol)


def kelly_from_returns(returns, fraction: float = 1.0) -> float:
    """新增能力：从交易收益序列估计连续凯利分数 f = E[r] / Var[r]。

    仅依赖样本均值/方差，比手工填胜率/赔率更数据驱动；fraction 支持分数凯利。
    收益不足 2 个有限观测或方差<=0 时返回 0（无信息不下注）。
    """
    try:
        fr = float(fraction)
    except (TypeError, ValueError):
        return 0.0
    if not (0.0 < fr <= 1.0):
        raise ValueError(f"fraction 必须介于 (0,1]，收到 {fraction!r}")
    r = pd.Series(returns).astype(float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mu = float(r.mean())
    var = float(r.var(ddof=1))
    if var <= 0:
        return 0.0
    f = mu / var
    return float(max(0.0, min(1.0, f * fr)))
