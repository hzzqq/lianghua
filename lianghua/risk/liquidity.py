"""流动性风险：Amihud 非流动性指标 + 换手冲击成本估计（迭代打磨）。

纯 pandas/numpy。

迭代打磨（本轮回测收尾）：
- 新增 amihud_series()：返回逐期 Amihud 非流动性序列（|收益|/成交额），用于
  分布分析，amihud() 改为取其均值（可观测、可诊断）。
- 隐性修复：原 amihud 对全 NaN 输入静默返回 NaN.mean()=NaN 无提示；
  现显式校验输入为 Series、对齐、非负成交量/价格，全非法时抛错。
- liquidity_cost 修复隐性 bug：原 weight 参数完全未被使用，且未知 ADV 时
  用魔法常数 10.0 相乘（语义不明）；现成本随权重与参与度平方根缩放，
  未知 ADV 按最坏参与度上限估计，参数可调。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["amihud", "amihud_series", "liquidity_cost"]


def _check_triple(returns: pd.Series, volume: pd.Series, price: pd.Series) -> pd.DataFrame:
    for name, s in (("returns", returns), ("volume", volume), ("price", price)):
        if not isinstance(s, pd.Series):
            raise TypeError(f"{name} 必须是 pandas.Series")
    # 用 inner join 显式取三者的共同观测：原先的 outer join 会先并集再靠下面的
    # 有限性过滤把多出来的 NaN 行丢掉，结果完全相同，但 pandas 对"全 DatetimeIndex
    # 的并集是否排序"已标记弃用（未来默认不排序 → 时间序列会变成非时序），
    # inner join 既避开这个未来的静默行为变化，也更贴合本函数的语义。
    df = pd.concat(
        [returns, volume, price], axis=1, keys=["r", "v", "p"], join="inner"
    ).astype(float)
    # 隐性修复：原对 NaN/inf 不校验，污染 |r|/成交额
    df = df[np.isfinite(df).all(axis=1)]
    if df.empty:
        raise ValueError("无共同有限观测，无法计算 Amihud 指标")
    if (df["v"] < 0).any() or (df["p"] < 0).any():
        raise ValueError("volume/price 不可为负")
    return df


def amihud_series(returns: pd.Series, volume: pd.Series, price: pd.Series) -> pd.Series:
    """新增能力：逐期 Amihud 非流动性序列 = |收益| / (成交量×价格)。

    零成交额日记为 NaN（避免除零 inf），返回与输入对齐的 Series。
    """
    df = _check_triple(returns, volume, price)
    dv = (df["v"] * df["p"]).replace(0, np.nan)
    illiq = (df["r"].abs() / dv).replace([np.inf, -np.inf], np.nan)
    return illiq


def amihud(returns: pd.Series, volume: pd.Series, price: pd.Series) -> float:
    """Amihud 非流动性 = |收益| / 成交额 的均值（越大越难成交）。"""
    s = amihud_series(returns, volume, price)
    if s.isna().all():
        raise ValueError("所有观测成交额为零，无法估计 Amihud 指标")
    return float(s.mean())


def liquidity_cost(weights: dict, adv: dict, participant_rate: float = 0.1,
                   impact_k: float = 0.1) -> pd.DataFrame:
    """估计各资产市场冲击成本（% 成交金额）。

    weights: {资产: 权重(0~1)}；adv: {资产: 日均成交量}；
    participant_rate: 单次成交占 ADV 比例（默认 10%）；
    impact_k: 冲击系数（平方根冲击模型 cost% = k·√participation·|weight|）。

    隐性修复：原实现 weight 完全未被使用；未知 ADV 按最坏参与度(100%)估计。
    """
    if not (0.0 < participant_rate <= 1.0):
        raise ValueError("participant_rate 必须介于 (0,1]")
    if impact_k < 0:
        raise ValueError("impact_k 必须 >= 0")
    rows = []
    for a, w in weights.items():
        adv_a = adv.get(a)
        part = 1.0 if (adv_a is None or adv_a <= 0) else min(participant_rate, 1.0)
        cost = impact_k * np.sqrt(part) * abs(float(w))
        rows.append({"资产": a, "权重": float(w), "冲击成本%": float(cost * 100)})
    return pd.DataFrame(rows)
