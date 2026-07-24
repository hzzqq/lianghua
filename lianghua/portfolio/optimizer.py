"""组合优化器：等权 / 风险平价 / 均值方差 / Kelly。

输入：各资产(或策略)的**日收益矩阵** DataFrame，每列一个标的。
输出：权重 Series（和为 1，默认非负）。全部纯 pandas/numpy。

v2 改进：
- 新增 risk_contributions()：各资产对组合方差的风险贡献占比（可观测/诊断）。
- optimize() 支持 max_weight 集中度上限（迭代投影），避免单资产过度集中。
- 统一输入守卫：非 DataFrame / 空 / 资产数不足 / 收益行数不足会抛清晰错误，
  杜绝隐性 NaN 权重；逆波动对零波动列优雅回退。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns 必须是 DataFrame（每列一个资产，行=日收益）")
    if returns.shape[1] == 0:
        raise ValueError("returns 至少需要一个资产列")
    if returns.shape[0] < 2:
        raise ValueError("returns 至少需要 2 行收益才能估计协方差/波动")
    return returns.dropna(how="any")


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """等权配置。"""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns 必须是 DataFrame（每列一个资产，行=日收益）")
    if returns.shape[1] == 0:
        raise ValueError("returns 至少需要一个资产列")
    n = returns.shape[1]
    return pd.Series(np.ones(n) / n, index=returns.columns)


def inverse_vol(returns: pd.DataFrame) -> pd.Series:
    """逆波动加权（风险平价的常用近似）：w_i ∝ 1/σ_i。"""
    r = _check_returns(returns)
    vol = r.std(ddof=1)
    inv = 1.0 / vol.replace(0, np.nan)
    # 零/缺失波动（如常数列）回退为该列均值，避免 NaN 权重
    if inv.isna().any():
        fill = inv.replace([np.inf, -np.inf], np.nan).dropna().mean()
        inv = inv.replace([np.inf, -np.inf], np.nan).fillna(fill if pd.notna(fill) else 1.0)
    inv = np.maximum(inv, 0.0)
    if inv.sum() <= 0:
        return equal_weight(returns)
    return inv / inv.sum()


def risk_parity(returns: pd.DataFrame, max_iter: int = 500, tol: float = 1e-8) -> pd.Series:
    """迭代法风险平价：各资产边际风险贡献趋近相等。"""
    r = _check_returns(returns)
    if r.shape[1] == 1:
        return pd.Series([1.0], index=returns.columns)
    cov = r.cov().values
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        sw = cov @ w
        # 固定点更新 w_i ∝ 1 / (Σw)_i
        new_w = 1.0 / np.maximum(sw, 1e-12)
        new_w = new_w / new_w.sum()
        if np.max(np.abs(new_w - w)) < tol:
            w = new_w
            break
        w = new_w
    return pd.Series(w, index=returns.columns)


def _mv_weights(returns: pd.DataFrame, risk_aversion: float) -> pd.Series:
    r = _check_returns(returns)
    mu = r.mean().values * 252.0           # 年化期望
    cov = r.cov().values * 252.0           # 年化协方差
    try:
        inv = np.linalg.pinv(cov)
        raw = inv @ mu
    except np.linalg.LinAlgError:
        raw = np.ones_like(mu)
    # 非负约束：简单投影到非负再归一化（务实近似，避免外部依赖）
    raw = np.clip(raw, 0, None)
    if raw.sum() <= 0:
        raw = np.ones_like(mu)
    w = raw / raw.sum()
    return pd.Series(w, index=returns.columns)


def mean_variance(returns: pd.DataFrame) -> pd.Series:
    """均值方差（不允许做空，非负权重）。"""
    return _mv_weights(returns, risk_aversion=1.0)


def kelly(returns: pd.DataFrame) -> pd.Series:
    """多资产 Kelly 配置（风险厌恶=1 的均值方差）。"""
    return _mv_weights(returns, risk_aversion=1.0)


def risk_contributions(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """各资产对组合方差的风险贡献占比（非负权重下和为 1）。

    贡献_i = w_i * (Σw)_i / (wᵀΣw)。用于诊断集中度——理想风险平价下各贡献应相等。
    """
    w = np.asarray(weights, dtype=float)
    r = returns.dropna(how="any")
    if r.shape[0] < 2 or w.sum() == 0:
        return pd.Series(np.nan, index=returns.columns)
    cov = r.cov().values
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return pd.Series(np.nan, index=returns.columns)
    mrc = cov @ w
    contrib = (w * mrc) / port_var
    return pd.Series(contrib, index=returns.columns)


def _apply_max_weight(w: np.ndarray, cap: float | None) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    total = w.sum()
    if total > 0:
        w = w / total
    if cap is None:
        return w
    cap = float(cap)
    for _ in range(100):
        w = np.minimum(w, cap)
        s = w.sum()
        if s <= 0:
            break
        w = w / s
        if np.all(w <= cap + 1e-9):
            break
    return w


def optimize(returns: pd.DataFrame, method: str = "risk_parity",
             max_weight: float | None = None) -> pd.Series:
    """统一入口。method ∈ {equal, inverse_vol, risk_parity, mean_variance, kelly}。

    max_weight: 单资产权重上限（0~1），超出部分迭代再分配（集中度控制）。
    """
    methods = {
        "equal": equal_weight,
        "inverse_vol": inverse_vol,
        "risk_parity": risk_parity,
        "mean_variance": mean_variance,
        "kelly": kelly,
    }
    if method not in methods:
        raise ValueError(f"未知方法 {method}，可选 {list(methods)}")
    w = methods[method](returns)
    if max_weight is not None:
        if not (0 < max_weight < 1):
            raise ValueError("max_weight 必须在 (0, 1) 区间")
        w = pd.Series(_apply_max_weight(w.values, max_weight), index=w.index)
    return w


def portfolio_returns(weights: pd.Series, asset_returns: pd.DataFrame) -> pd.Series:
    """由权重与资产日收益得到组合日收益序列。"""
    aligned = asset_returns.reindex(columns=weights.index).fillna(0.0)
    return (aligned * weights).sum(axis=1)
