"""高级组合优化器（迭代 121-125）：
max_sharpe / min_cvar / shrinkage_min_var(Ledoit-Wolf) / max_entropy / momentum_score。

统一约定：输入日频收益率 DataFrame（列=资产），输出对齐列名的权重 pd.Series（和≈1、非负，
个别方法允许缩放，编排层会再归一）。纯 pandas/numpy + 轻量投影，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _simplex_project(w: np.ndarray) -> np.ndarray:
    """投影到单纯形（非负、和为1），用于无约束解的清洗。"""
    w = np.asarray(w, dtype=float)
    if not np.isfinite(w).all():
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(w)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not cond.any():
        return np.ones(n) / n
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    out = np.maximum(w - theta, 0.0)
    s = out.sum()
    return out / s if s > 0 else np.ones(n) / n


def sanitize_weights(w, index, long_only: bool = True, eps: float = 1e-12) -> np.ndarray:
    """把任意权重数组清洗为合法组合权重：非有限→0、可选只多空、归一。

    - 非有限值（NaN/±inf）先清零，避免污染下游；
    - long_only=True 时截断负权重（不允许做空）；
    - 归一化为和=1；若全为 0（或退化），退化为等权，保证可下单。
    """
    w = np.asarray(w, dtype=float).ravel()
    if w.shape[0] != len(index):
        raise ValueError(f"权重长度 {w.shape[0]} 与资产数 {len(index)} 不一致")
    if not np.isfinite(w).all():
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if long_only:
        w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= eps:
        return np.ones(len(index)) / len(index)
    return w / s


def _finalize(w, index) -> pd.Series:
    """优化器统一出口：清洗 + 对齐索引。"""
    return pd.Series(sanitize_weights(w, index), index=index)


def max_sharpe(returns: pd.DataFrame, rf: float = 0.0, annual: int = 252) -> pd.Series:
    """切点组合（最大化夏普）：w ∝ Σ⁻¹(μ-rf)，再投影到单纯形。"""
    mu = returns.mean().values * annual - rf
    cov = returns.cov().values * annual
    try:
        raw = np.linalg.solve(cov + np.eye(len(mu)) * 1e-8, mu)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cov) @ mu
    return _finalize(_simplex_project(raw), returns.columns)


def min_cvar(returns: pd.DataFrame, alpha: float = 0.05, n_iter: int = 400,
             seed: int = 0) -> pd.Series:
    """最小化组合 CVaR（历史情景，随机搜索 + 单纯形投影的轻量近似）。"""
    R = returns.values
    m = R.shape[1]
    rng = np.random.default_rng(seed)

    def cvar(w):
        pnl = R @ w
        var = np.quantile(pnl, alpha)
        tail = pnl[pnl <= var]
        return -(tail.mean() if len(tail) else var)

    best_w = np.ones(m) / m
    best = cvar(best_w)
    for _ in range(n_iter):
        cand = _simplex_project(best_w + rng.normal(0, 0.15, m))
        val = cvar(cand)
        if val < best:
            best, best_w = val, cand
    return _finalize(best_w, returns.columns)


def shrinkage_min_var(returns: pd.DataFrame, shrink: float = 0.2,
                      annual: int = 252) -> pd.Series:
    """Ledoit-Wolf 风格协方差收缩后的最小方差组合。

    Σ_shrunk = (1-δ)·S + δ·F，F 为对角常数方差目标（抑制估计噪声）。
    """
    S = returns.cov().values * annual
    n = S.shape[0]
    mu_var = np.trace(S) / n
    F = np.eye(n) * mu_var
    Sig = (1 - shrink) * S + shrink * F
    ones = np.ones(n)
    try:
        inv = np.linalg.inv(Sig + np.eye(n) * 1e-10)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sig)
    raw = inv @ ones
    return _finalize(_simplex_project(raw), returns.columns)


def max_entropy(returns: pd.DataFrame) -> pd.Series:
    """最大分散熵：以逆波动为先验、抑制集中度，权重 ∝ 1/vol 再归一。

    等价于在无相关性假设下最大化 -Σ w_i ln w_i 约束下的风险均衡近似。
    """
    vol = returns.std().values
    inv = 1.0 / (vol + 1e-12)
    w = inv / inv.sum()
    return _finalize(w, returns.columns)


def momentum_score(returns: pd.DataFrame, lookback: int = 60,
                   top: int = 0) -> pd.Series:
    """动量打分配权：按近 lookback 累计收益排序，正动量归一配权。

    top>0 时只保留分数最高的 top 个资产（其余置 0）。
    """
    look = returns.iloc[-lookback:] if len(returns) > lookback else returns
    score = (1 + look).prod() - 1
    score = score.clip(lower=0.0)
    if top and top < len(score):
        keep = score.nlargest(top).index
        score = score.where(score.index.isin(keep), 0.0)
    s = score.sum()
    if s <= 0:
        return _finalize(np.ones(len(score)) / len(score), returns.columns)
    w = (score / s).reindex(returns.columns).fillna(0.0).values
    return _finalize(w, returns.columns)


def min_tail_risk(returns: pd.DataFrame, alpha: float = 0.05,
                  n_iter: int = 400, seed: int = 0) -> pd.Series:
    """最小尾部风险：随机搜索最小化「CVaR + 0.5×最差单期损失」。

    比 min_cvar 更强调极端单点亏损（而非仅尾部均值），对跳空/崩盘更稳健。
    权重投影到单纯形（非负、和为1）。
    """
    R = returns.values
    m = R.shape[1]
    rng = np.random.default_rng(seed)

    def tail_risk(w):
        pnl = R @ w
        var = np.quantile(pnl, alpha)
        tail = pnl[pnl <= var]
        cvar = -(tail.mean() if len(tail) else var)
        worst = -pnl.min()  # 最差单期亏损（越大越糟）
        return cvar + 0.5 * worst

    best_w = np.ones(m) / m
    best = tail_risk(best_w)
    for _ in range(n_iter):
        cand = _simplex_project(best_w + rng.normal(0, 0.15, m))
        val = tail_risk(cand)
        if val < best:
            best, best_w = val, cand
    return _finalize(best_w, returns.columns)


def vol_target_opt(returns: pd.DataFrame, target_vol: float = 0.12,
                   annual: int = 252) -> pd.Series:
    """目标波动率权重（近似）：以逆波动为先验，缩放使组合波动≈target，再投影到单纯形。

    解析上多空/带杠杆可精确命中 target；本平台约束为多头且和=1，
    故采用「逆波动先验 × 缩放系数 → 单纯形投影」的近似方案。
    """
    R = returns.values
    n = R.shape[1]
    vol = returns.std().values * np.sqrt(annual)
    inv = 1.0 / (vol + 1e-12)
    base = inv / inv.sum()
    cov = returns.cov().values * annual
    port_vol = np.sqrt(max(base @ cov @ base, 1e-12))
    scale = target_vol / port_vol
    raw = base * scale
    return _finalize(_simplex_project(raw), returns.columns)


def bayesian_shrinkage(returns: pd.DataFrame, delta: float = 0.2,
                        annual: int = 252) -> pd.Series:
    """贝叶斯收缩协方差后的最小方差组合。

    把样本协方差 S（n_obs 个观测）视作似然，向「各资产等方差、零相关的
    对角先验 F」做贝叶斯更新：Σ_post = (n_obs·S + n_prior·F)/(n_obs+n_prior)，
    其中 n_prior = delta·n_obs。再解最小方差（Σ⁻¹·1 投影到单纯形）。
    """
    R = returns.values
    n_obs = R.shape[0]
    S = returns.cov().values * annual
    n = S.shape[0]
    mu_var = np.trace(S) / n
    F = np.eye(n) * mu_var
    n_prior = max(delta, 1e-3) * n_obs
    Sig = (n_obs * S + n_prior * F) / (n_obs + n_prior)
    ones = np.ones(n)
    try:
        inv = np.linalg.inv(Sig + np.eye(n) * 1e-10)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sig)
    raw = inv @ ones
    return _finalize(_simplex_project(raw), returns.columns)


def max_return(returns: pd.DataFrame, annual: int = 252) -> pd.Series:
    """最大化预期收益配置：以正预期收益为先验、投影到单纯形（仅做多正动量资产）。

    若全部资产预期收益非正，退化为等权。
    """
    mu = returns.mean().values * annual
    raw = np.clip(mu, 0.0, None)
    if raw.sum() <= 0:
        raw = np.ones_like(mu)
    return _finalize(_simplex_project(raw), returns.columns)


def min_track_error(returns: pd.DataFrame, bench: list | np.ndarray | None = None) -> pd.Series:
    """最小跟踪误差配置：使组合相对基准 b 的主动收益方差最小。

    解析上最优解即 w = b（完全贴合基准）。b 缺省为等权基准。
    bench 长度必须与资产数一致，否则明确报错而非静默错位。
    """
    n = returns.shape[1]
    if bench is None:
        b = np.ones(n) / n
    else:
        b = np.asarray(bench, dtype=float).ravel()
        if b.shape[0] != n:
            raise ValueError(f"bench 长度 {b.shape[0]} 与资产数 {n} 不一致")
        s = b.sum()
        b = b / s if s > 0 else np.ones(n) / n
    return _finalize(b, returns.columns)


def robust_cov(returns: pd.DataFrame, delta: float = 0.3,
                annual: int = 252) -> pd.Series:
    """稳健协方差最小方差：更强收缩 + 岭正则，抑制小样本估计噪声。

    比 shrinkage_min_var 收缩更重，对高相关/极端样本更稳健。
    """
    R = returns.values
    S = returns.cov().values * annual
    n = S.shape[0]
    mu_var = np.trace(S) / n
    F = np.eye(n) * mu_var
    Sig = (1 - delta) * S + delta * F
    ones = np.ones(n)
    try:
        inv = np.linalg.inv(Sig + np.eye(n) * 1e-8)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sig)
    raw = inv @ ones
    return _finalize(_simplex_project(raw), returns.columns)


def compare_optimizers(returns: pd.DataFrame, names: list | None = None) -> pd.DataFrame:
    """对给定收益矩阵批量运行注册表优化器，返回 资产×优化器 的权重对比矩阵。

    单个优化器失败（如该收益样本不满足其假设）会被隔离跳过，不影响其余对比；
    便于研究/UI 一键横向比较多种配置。新增能力（超越单个 optimize_weights）。
    """
    from .registry import OPTIMIZER_REGISTRY, OPTIMIZER_NAMES

    names = names or OPTIMIZER_NAMES
    out = {}
    for n in names:
        if n not in OPTIMIZER_REGISTRY:
            continue
        try:
            w = OPTIMIZER_REGISTRY[n](returns)
            out[n] = pd.Series(w).reindex(returns.columns).fillna(0.0)
        except Exception:
            # 隔离单个优化器的失败，保证对比表可用
            continue
    return pd.DataFrame(out, index=returns.columns)
