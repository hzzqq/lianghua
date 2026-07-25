"""VaR / CVaR 风险度量（迭代22）。

零额外依赖（仅 pandas/numpy）：
- historical_var  : 历史模拟法 VaR
- parametric_var  : 参数法（正态）VaR
- cvar            : 条件 VaR / 期望损失 (Expected Shortfall)
- var_report      : 一站式风险报告（多置信度）

约定：输入为**收益率序列**（如 equity.pct_change().dropna()），
输出为正数表示损失幅度（如 0.025 = 2.5% 潜在亏损）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["historical_var", "parametric_var", "cvar", "monte_carlo_var", "var_report"]

# 常用置信度对应的标准正态分位数（避免依赖 scipy）
_Z = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263, 0.995: 2.5758}


def _z_score(conf: float) -> float:
    """置信度 -> 正态分位数。非常用值用 Acklam 逆 CDF 近似。"""
    if not (0.0 < conf < 1.0):
        raise ValueError("conf 必须介于 0 与 1 之间")
    if conf in _Z:
        return _Z[conf]
    # Peter Acklam 近似逆正态 CDF
    p = 1 - conf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    q = np.sqrt(-2 * np.log(p))
    z = (((((a[0] * q + a[1]) * q + a[2]) * q + a[3]) * q + a[4]) * q + a[5]) / \
        ((((b[0] * q + b[1]) * q + b[2]) * q + b[3]) * q + b[4] * q / q)
    return abs(z)


def _clean(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        raise ValueError("收益率序列太短，至少需要 2 个观测值")
    return r


def historical_var(returns: pd.Series, conf: float = 0.95) -> float:
    """历史模拟法 VaR：损失分布的 (1-conf) 分位数。"""
    r = _clean(returns)
    return float(max(0.0, -np.quantile(r, 1 - conf)))


def parametric_var(returns: pd.Series, conf: float = 0.95) -> float:
    """参数法（正态假设）VaR = -(mu - z * sigma)。"""
    r = _clean(returns)
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    return float(max(0.0, -(mu - _z_score(conf) * sigma)))


def cvar(returns: pd.Series, conf: float = 0.95) -> float:
    """条件 VaR（期望损失）：损失超过 VaR 部分的平均值。"""
    if not (0.0 < conf < 1.0):
        raise ValueError("conf 必须介于 0 与 1 之间")
    r = _clean(returns)
    var = -np.quantile(r, 1 - conf)
    tail = r[r <= -var]
    if len(tail) == 0:
        return float(max(0.0, var))
    return float(max(0.0, -tail.mean()))


def monte_carlo_var(returns: pd.Series, conf: float = 0.95,
                    n_sims: int = 10000, method: str = "bootstrap",
                    seed: int | None = None) -> float:
    """蒙特卡洛 VaR（新增能力）。

    method:
    - "bootstrap": 从历史收益有放回重采样（不依赖正态假设，最稳健）。
    - "normal":   按样本均值/标准差做正态模拟。

    seed 可复现。返回正数损失幅度。隐性修复：conf/ n_sims 非法时显式抛错，
    而非让 np.random / quantile 给出无意义值。
    """
    if not (0.0 < conf < 1.0):
        raise ValueError("conf 必须介于 0 与 1 之间")
    if n_sims <= 0:
        raise ValueError("n_sims 必须为正整数")
    r = _clean(returns)
    rng = np.random.default_rng(seed)
    if method == "bootstrap":
        draws = r.to_numpy()
        idx = rng.integers(0, len(draws), size=n_sims)
        sim = draws[idx]
    elif method == "normal":
        mu, sigma = float(r.mean()), float(r.std(ddof=1))
        if sigma <= 0:
            sigma = 1e-12  # 退化标准差兜底，避免全 0 模拟
        sim = rng.normal(mu, sigma, size=n_sims)
    else:
        raise ValueError("method 必须是 'bootstrap' 或 'normal'")
    return float(max(0.0, -np.quantile(sim, 1 - conf)))


def var_report(returns: pd.Series,
               confs: tuple = (0.90, 0.95, 0.99),
               horizon_days: int = 1,
               notional: float | None = None) -> pd.DataFrame:
    """多置信度风险报告。

    horizon_days: 持有期（按 sqrt(T) 缩放）
    notional: 若给出名义本金，额外输出金额列
    """
    r = _clean(returns)
    if horizon_days < 1:
        raise ValueError("horizon_days 必须 >= 1")
    for c in confs:
        if not (0.0 < c < 1.0):
            raise ValueError(f"置信度 {c} 必须介于 0 与 1 之间")
    if notional is not None and notional <= 0:
        raise ValueError("notional 必须为正")
    scale = np.sqrt(horizon_days)
    rows = []
    for c in confs:
        hv = historical_var(r, c) * scale
        pv = parametric_var(r, c) * scale
        cv = cvar(r, c) * scale
        row = {"置信度": c, "历史VaR": hv, "参数VaR": pv, "CVaR": cv}
        if notional:
            row["历史VaR金额"] = hv * notional
            row["CVaR金额"] = cv * notional
        rows.append(row)
    return pd.DataFrame(rows)
