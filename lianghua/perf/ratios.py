"""进阶绩效比率（迭代 126-128）：
ulcer_index / martin_ratio / gain_to_pain / cagr / up_down_capture / k_ratio。

输入统一为净值曲线 equity(pd.Series)；捕获比率额外需要基准净值。
纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _returns(equity: pd.Series) -> pd.Series:
    return equity.astype(float).pct_change().dropna()


def ulcer_index(equity: pd.Series) -> float:
    """溃疡指数：回撤深度的均方根，越小越稳。"""
    e = equity.astype(float)
    peak = e.cummax()
    dd = (e / peak - 1.0) * 100
    return float(np.sqrt((dd ** 2).mean()))


def martin_ratio(equity: pd.Series, rf: float = 0.0, annual: int = 252) -> float:
    """Martin 比率(溃疡表现指数)：年化超额收益 / 溃疡指数。"""
    r = _returns(equity)
    if len(r) == 0:
        return 0.0
    ann_ret = (1 + r.mean()) ** annual - 1 - rf
    ui = ulcer_index(equity)
    return float(ann_ret / (ui / 100 + 1e-12))


def gain_to_pain(equity: pd.Series) -> float:
    """收益痛苦比：正收益之和 / 负收益绝对值之和。"""
    r = _returns(equity)
    gains = r[r > 0].sum()
    pains = -r[r < 0].sum()
    return float(gains / (pains + 1e-12))


def cagr(equity: pd.Series, annual: int = 252) -> float:
    """年化复合增长率（按数据点数折算）。"""
    e = equity.astype(float)
    if len(e) < 2 or e.iloc[0] <= 0 or annual <= 0:
        return 0.0
    total = e.iloc[-1] / e.iloc[0]
    # 净值跌破/归零时 total<=0，负底数做分数幂会抛 ValueError；此处安全降级为 0.0
    if total <= 0:
        return 0.0
    years = len(e) / annual
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def up_down_capture(equity: pd.Series, benchmark: pd.Series) -> dict:
    """上/下行捕获比：基准上涨/下跌时组合的相对表现。"""
    r = _returns(equity)
    b = _returns(benchmark)
    idx = r.index.intersection(b.index)
    r, b = r.loc[idx], b.loc[idx]
    up = b > 0
    dn = b < 0
    up_cap = (r[up].mean() / (b[up].mean() + 1e-12)) if up.any() else 0.0
    dn_cap = (r[dn].mean() / (b[dn].mean() + 1e-12)) if dn.any() else 0.0
    return {"up_capture": float(up_cap), "down_capture": float(dn_cap)}


def k_ratio(equity: pd.Series) -> float:
    """K 比率：对数净值线性回归斜率 / 斜率标准误，衡量收益的稳定性/一致性。"""
    e = equity.astype(float)
    e = e[e > 0]
    if len(e) < 3:
        return 0.0
    y = np.log(e.values)
    x = np.arange(len(y), dtype=float)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / (sxx + 1e-12)
    resid = y - (ym + slope * (x - xm))
    dof = max(len(y) - 2, 1)
    se = np.sqrt((resid ** 2).sum() / dof / (sxx + 1e-12))
    return float(slope / (se + 1e-12))


def _ensure_returns(x: pd.Series) -> pd.Series:
    """疑似净值曲线(多数点>1.5)则转收益率，否则当收益率用。"""
    x = x.astype(float)
    if (x > 1.5).mean() > 0.5:
        return x.pct_change().dropna()
    return x


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Omega 比率：相对阈值的正收益期望 / 负收益期望（全分布信息）。"""
    r = _ensure_returns(returns)
    if len(r) == 0:
        return 0.0
    gain = r[r > threshold].sub(threshold).sum()
    loss = -(r[r < threshold].sub(threshold)).sum()
    return float(gain / (loss + 1e-12))


def calmar_ratio(equity: pd.Series, annual: int = 252) -> float:
    """Calmar 比率：年化收益 / |最大回撤|。"""
    e = equity.astype(float)
    if len(e) < 2 or e.iloc[0] <= 0:
        return 0.0
    years = len(e) / annual
    if years <= 0:
        return 0.0
    c = cagr(e, annual)
    mdd = (e / e.cummax() - 1.0).min()
    return float(c / (-mdd + 1e-12))


def tail_ratio(returns) -> float:
    """尾部比：右尾均值(95%分位以上) / 左尾均值绝对值(5%分位以下)。>1 右尾更肥。"""
    r = _ensure_returns(returns)
    if len(r) < 20:
        return 0.0
    right = r[r > r.quantile(0.95)].mean()
    left = -r[r < r.quantile(0.05)].mean()
    return float(right / (left + 1e-12))


def summary(equity: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """一次性汇总常用绩效比率，所有值经安全化（无 NaN/inf），可直接落库/出报告。

    新需求：把分散的比率函数聚合成一份「可观测、可序列化」的绩效快照。
    含 CAGR / 年化波动 / 最大回撤 / Calmar / 溃疡 / 收益痛苦比 / 尾部比，
    给定基准时附上/下行捕获比。所有除零均经 +1e-12 或显式守卫，避免污染下游。
    """
    e = equity.astype(float)
    r = _returns(e)
    ann_vol = float(r.std() * np.sqrt(252)) if len(r) else 0.0
    peak = e.cummax()
    mdd = float((e / peak - 1.0).min())
    out = {
        "cagr": cagr(e),
        "annual_vol": ann_vol,
        "max_drawdown": mdd,
        "calmar": calmar_ratio(e),
        "ulcer_index": ulcer_index(e),
        "gain_to_pain": gain_to_pain(e),
        "tail_ratio": tail_ratio(r),
        "k_ratio": k_ratio(e),
    }
    if benchmark is not None:
        cap = up_down_capture(e, benchmark)
        out["up_capture"] = cap["up_capture"]
        out["down_capture"] = cap["down_capture"]
    # 安全清洗：任何 NaN/inf 回落为 0.0，保证序列化安全
    for k, v in out.items():
        if not np.isfinite(v):
            out[k] = 0.0
    return out
