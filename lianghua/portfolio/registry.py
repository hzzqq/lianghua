"""组合优化器 / 权重生成器 统一注册表（迭代11-100 收口）。

把散落在 portfolio/ 与 strategy/ 的「吃收益矩阵 -> 产出权重」的函数，
统一成 ``name -> callable(returns_df, **kw) -> weight Series`` 的入口，
供 orchestrator / UI / CLI 一键调用。

仅收录「输出可直接作为组合权重的函数」；cppi(组合净值)、clustering(聚类标签)、
pair/kalman(双资产对冲比) 等形态不同，列入 master 能力索引做发现，不进本权重表。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- 延迟导入，避免循环 -------------------------------------------------------
from .optimizer import (
    equal_weight, inverse_vol, risk_parity, mean_variance, kelly, optimize,
)
from .min_variance import min_variance
from .max_diversification import max_diversification
from .hrp import hrp
from .target_vol import target_vol_weights
from .risk_parity_ewm import risk_parity_ewm
from .allweather import all_weather_weights
from ..strategy.rotation import rotation_weights
from ..strategy.dual_momentum import dual_momentum
from ..risk.budget import equal_risk_contribution as _erc, target_risk_budget as _trb
from .advanced import (
    max_sharpe, min_cvar, shrinkage_min_var, max_entropy, momentum_score,
    min_tail_risk, vol_target_opt,
    bayesian_shrinkage, max_return, min_track_error, robust_cov,
    sanitize_weights,
)


def _cov(returns: pd.DataFrame, annual: int = 252) -> np.ndarray:
    return returns.cov().values * annual


def _as_series(w, index):
    """ndarray / Series / DataFrame 统一成一行权重 Series。"""
    if isinstance(w, pd.DataFrame):
        w = w.iloc[-1]
    if isinstance(w, pd.Series):
        s = w.reindex(index).fillna(0.0)
    else:
        s = pd.Series(w, index=index)
    return s


def _dm_wrap(returns: pd.DataFrame, **kw):
    """dual_momentum 返回"每时点持哪个资产"的标签序列，这里取末次持仓转 one-hot 权重。"""
    pos = dual_momentum(returns, lookback=kw.get("lookback", 120))
    pos = pd.Series(pos).dropna()
    w = pd.Series(0.0, index=returns.columns)
    if len(pos):
        last = pos.iloc[-1]
        if last in w.index:
            w[last] = 1.0
    return w


def _bl_wrap(returns: pd.DataFrame, **kw):
    from .black_litterman import black_litterman
    mu = returns.mean() * 252
    cov = _cov(returns)
    n = len(mu)
    pi = mu.values.astype(float)
    P = np.eye(n)
    Q = pi.copy()
    w = black_litterman(pi, cov, P, Q)
    s = w.sum()
    if s != 0:
        w = w / s
    return pd.Series(w, index=mu.index)


def _trb_wrap(returns: pd.DataFrame, **kw):
    cov = _cov(returns)
    n = cov.shape[0]
    budgets = kw.get("budgets", np.ones(n) / n)
    w = _trb(cov, np.asarray(budgets))
    s = w.sum()
    if s != 0:
        w = w / s
    return pd.Series(w, index=returns.columns)


# name -> (callable, description)
_BUILD = {
    "equal_weight":       (lambda r, **k: equal_weight(r), "等权"),
    "inverse_vol":        (lambda r, **k: inverse_vol(r), "逆波动率"),
    "risk_parity":        (lambda r, **k: risk_parity(r), "风险平价"),
    "mean_variance":      (lambda r, **k: mean_variance(r), "均值方差"),
    "kelly":              (lambda r, **k: kelly(r), "凯利"),
    "optimize":           (lambda r, **k: optimize(r, method=k.get("method", "risk_parity")), "统一入口(可选方法)"),
    "min_variance":       (lambda r, **k: _as_series(min_variance(_cov(r)), r.columns), "最小方差"),
    "max_diversification":(lambda r, **k: _as_series(max_diversification(_cov(r)), r.columns), "最大分散化"),
    "hrp":                (lambda r, **k: _as_series(hrp(_cov(r)), r.columns), "层次风险平价(HRP)"),
    "target_vol":         (lambda r, **k: target_vol_weights(r, target_vol=k.get("target_vol", 0.12)), "目标波动缩放"),
    "risk_parity_ewm":    (lambda r, **k: risk_parity_ewm(r, span=k.get("span", 60)), "EWM 风险平价"),
    "equal_risk_contrib": (lambda r, **k: _as_series(_erc(_cov(r)), r.columns), "等风险贡献(ERC)"),
    "target_risk_budget": (_trb_wrap, "目标风险预算"),
    "all_weather":        (lambda r, **k: all_weather_weights(r, method=k.get("method", "risk_parity")), "全天候"),
    "rotation":           (lambda r, **k: _as_series(rotation_weights(r, top=k.get("top", 1), lookback=k.get("lookback", 60)), r.columns), "动量轮动(前N)"),
    "dual_momentum":      (_dm_wrap, "双动量配置(末次持仓 one-hot)"),
    "black_litterman":    (_bl_wrap, "Black-Litterman(均衡视图)"),
    # 迭代 121-125：高级优化器
    "max_sharpe":         (lambda r, **k: max_sharpe(r, rf=k.get("rf", 0.0)), "最大夏普(切点组合)"),
    "min_cvar":           (lambda r, **k: min_cvar(r, alpha=k.get("alpha", 0.05)), "最小 CVaR"),
    "shrinkage_min_var":  (lambda r, **k: shrinkage_min_var(r, shrink=k.get("shrink", 0.2)), "收缩协方差最小方差(Ledoit-Wolf)"),
    "max_entropy":        (lambda r, **k: max_entropy(r), "最大分散熵"),
    "momentum_score":     (lambda r, **k: momentum_score(r, lookback=k.get("lookback", 60), top=k.get("top", 0)), "动量打分配权"),
    # 迭代 136-137：高级优化器
    "min_tail_risk":      (lambda r, **k: min_tail_risk(r, alpha=k.get("alpha", 0.05)), "最小尾部风险(CVaR+最差损失)"),
    "vol_target_opt":     (lambda r, **k: vol_target_opt(r, target_vol=k.get("target_vol", 0.12)), "目标波动率权重(近似)"),
    # 迭代 146-149：高级优化器
    "bayesian_shrinkage":(lambda r, **k: bayesian_shrinkage(r, delta=k.get("delta", 0.2)), "贝叶斯收缩最小方差"),
    "max_return":         (lambda r, **k: max_return(r), "最大化预期收益(正动量)"),
    "min_track_error":    (lambda r, **k: min_track_error(r, bench=k.get("bench", None)), "最小跟踪误差(贴合基准)"),
    "robust_cov":         (lambda r, **k: robust_cov(r, delta=k.get("delta", 0.3)), "稳健协方差最小方差(强收缩)"),
}


OPTIMIZER_REGISTRY: dict = {}
OPTIMIZER_INFO: dict = {}
for _n, (_fn, _d) in _BUILD.items():
    OPTIMIZER_REGISTRY[_n] = _fn
    OPTIMIZER_INFO[_n] = _d

OPTIMIZER_NAMES = list(OPTIMIZER_REGISTRY.keys())


def get_optimizer(name: str, **kwargs):
    """返回 ``f(returns_df, **kw) -> weight Series``。

    出口统一套一层可靠性护栏（sanitize_weights）：非有限值清零、长仓截断负权重、
    归一化为和=1；退化输入（协方差奇异/单资产/全零/含 NaN）无法求解析解时降级为等权，
    保证任何情况下都返回一个可下单的合法组合，绝不抛异常或吐出 NaN/inf。
    """
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"未知优化器: {name}，可选: {OPTIMIZER_NAMES}")
    fn = OPTIMIZER_REGISTRY[name]

    def _caller(returns: pd.DataFrame, **kw):
        kw.update(kwargs)
        try:
            w = fn(returns, **kw)
            return sanitize_weights(w, returns.columns)
        except Exception:  # noqa: BLE001
            # 任何内部异常（如协方差不可逆）都降级为等权，保证可下单
            n = len(returns.columns)
            return pd.Series(np.ones(n) / n, index=returns.columns)

    return _caller


def optimize_weights(name: str, returns: pd.DataFrame, **kwargs) -> pd.Series:
    """直接产出权重。"""
    return get_optimizer(name, **kwargs)(returns)
