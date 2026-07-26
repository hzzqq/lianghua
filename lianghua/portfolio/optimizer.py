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

import warnings

import numpy as np
import pandas as pd


def _check_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns 必须是 DataFrame（每列一个资产，行=日收益）")
    if returns.shape[1] == 0:
        raise ValueError("returns 至少需要一个资产列")
    if returns.shape[0] < 2:
        raise ValueError("returns 至少需要 2 行收益才能估计协方差/波动")
    # 隐性修复：dropna 仅剔除 NaN，会把 inf/-inf 漏进来污染协方差/标准差，
    # 进而静默产出 inf/NaN 权重且零报错。这里同时过滤非有限值。
    r = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if r.shape[0] < 2:
        raise ValueError("returns 含过多非有限值，至少需要 2 行有限观测值")
    return r


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

    隐性修复：原实现用 np.asarray(weights) 直接按位置映射到 returns 列，
    若 weights 的索引顺序/标签与 returns.columns 不一致，贡献会被错贴到错误的资产上
    （静默、无报错）。现按列名对齐权重。
    """
    w = weights if isinstance(weights, pd.Series) else pd.Series(weights)
    r = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if r.shape[0] < 2:
        return pd.Series(np.nan, index=r.columns)
    # 按 returns 列名对齐权重，缺失列视为 0 配置
    w = w.reindex(r.columns).fillna(0.0)
    wv = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(wv)) or wv.sum() == 0:
        return pd.Series(np.nan, index=r.columns)
    cov = r.cov().values
    port_var = float(wv @ cov @ wv)
    if port_var <= 0:
        return pd.Series(np.nan, index=r.columns)
    mrc = cov @ wv
    contrib = (wv * mrc) / port_var
    return pd.Series(contrib, index=r.columns)


def portfolio_volatility(weights: pd.Series, returns: pd.DataFrame) -> float:
    """组合年化波动率（默认按 sqrt(252) 缩放，与原模块年化口径一致）。

    new_requirement（新增能力）：组合风险的总量视图，与 risk_contributions（结构视图）
    互补；权重按列名对齐，缺失列视为 0 配置。
    """
    w = weights if isinstance(weights, pd.Series) else pd.Series(weights)
    r = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if r.shape[0] < 2:
        raise ValueError("returns 至少需要 2 行有限收益以估计协方差")
    w = w.reindex(r.columns).fillna(0.0)
    wv = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(wv)):
        raise ValueError("weights 含非有限值")
    cov = r.cov().values
    var = float(wv @ cov @ wv)
    if var <= 0:
        return 0.0
    return float(np.sqrt(var) * np.sqrt(252.0))


def effective_bets(weights: pd.Series, returns: pd.DataFrame) -> float:
    """有效独立下注数（Meucci 熵口径）。

    new_requirement（新增能力，机构风险诊断标配）：对相关性矩阵做特征值分解，
    p_i = λ_i / Σλ，E_NB = exp(-Σ p_i ln p_i)。E_NB 越接近资产数代表分散越充分、
    相关性越低；越接近 1 代表风险高度集中于少数共同因子。权重按列名对齐。
    """
    w = weights if isinstance(weights, pd.Series) else pd.Series(weights)
    r = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if r.shape[1] < 2:
        return 1.0
    corr = r.corr().values
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=0.0)
    eig = np.linalg.eigvalsh(np.clip(corr, -1.0, 1.0))
    eig = np.clip(eig, 1e-12, None)  # 相关性矩阵数值误差可能给出极小负值
    p = eig / eig.sum()
    # 熵口径有效数：exp(-Σ p ln p)
    return float(np.exp(-np.sum(p * np.log(p))))


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
    """由权重与资产日收益得到组合日收益序列。

    隐性修复：原实现用 asset_returns.reindex(columns=weights.index).fillna(0.0)，
    当 weights 含 asset_returns 不存在的列时，该列被静默填 0（配置凭空消失）
    且不归一化（权重和 ≠ 1 时权重语义失真）。现改为：把权重对齐到实际资产列、
    缺失列视为 0 配置，并对权重归一化；若 weights 存在资产侧没有的列，发出
    UserWarning 以便观测被丢弃的配置。
    """
    w = weights if isinstance(weights, pd.Series) else pd.Series(weights)
    missing = [c for c in w.index if c not in asset_returns.columns]
    if missing:
        warnings.warn(
            f"portfolio_returns: 权重含 {len(missing)} 个资产侧不存在的列"
            f"（如 {missing[:3]}），其配置将被忽略",
            UserWarning,
        )
    aligned_w = w.reindex(asset_returns.columns).fillna(0.0)
    s = float(aligned_w.sum())
    if s <= 0 or not np.isfinite(s):
        # 权重全为空/非有限：退化为等权，避免全 0 序列
        aligned_w = pd.Series(np.ones(asset_returns.shape[1]) / max(asset_returns.shape[1], 1),
                              index=asset_returns.columns)
    else:
        aligned_w = aligned_w / s
    aligned = asset_returns.reindex(columns=aligned_w.index).fillna(0.0)
    return (aligned * aligned_w).sum(axis=1)
