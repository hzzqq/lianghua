"""目标波动率组合：把组合波动缩放到目标水平。

隐性修复（本次）：原实现 asset_returns.astype(float).dropna() 不会剔除 inf，
inf 进入协方差后 port_vol 变 inf/NaN，scale = target_vol/inf 得到 0 或 NaN，
权重静默变成 0/NaN 且不报错。现统一过滤非有限值并做参数守卫；新增 max_leverage
上限防止"已实现波动趋零时杠杆无限放大"，并暴露 vol_target_scalar 复用缩放因子。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _port_annual_vol(returns: pd.DataFrame, w0: np.ndarray, periods: int) -> float:
    cov = returns.cov().values
    return float(np.sqrt(w0 @ cov @ w0) * np.sqrt(periods))


def target_vol_weights(asset_returns: pd.DataFrame, target_vol: float = 0.12,
                       periods: int = 252, max_leverage: float | None = None) -> pd.Series:
    """在等权基础上，按总波动反推使组合波动≈target_vol 的权重。

    max_leverage: 缩放因子(scale = target_vol/port_vol)上限，防止波动极低时杠杆失控。
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns 必须是 DataFrame（每列一个资产）")
    if not (target_vol > 0):
        raise ValueError("target_vol 必须为正")
    if periods < 1:
        raise ValueError("periods 必须为正整数")
    if max_leverage is not None and max_leverage <= 0:
        raise ValueError("max_leverage 必须为正")
    r = asset_returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if r.shape[1] == 0:
        raise ValueError("asset_returns 至少需要一个资产列")
    if r.shape[0] < 2:
        raise ValueError("asset_returns 至少需要 2 行有限收益")
    w0 = np.full(r.shape[1], 1.0 / r.shape[1])
    port_vol = _port_annual_vol(r, w0, periods)
    if not np.isfinite(port_vol) or port_vol <= 1e-12:
        # 波动趋零（如常数列）：无风险可缩放，回退等权而非放无限杠杆
        return pd.Series(w0, index=r.columns, name="weight")
    scale = target_vol / port_vol
    if max_leverage is not None:
        scale = min(scale, float(max_leverage))
    return pd.Series(w0 * scale, index=r.columns, name="weight")


def vol_target_scalar(asset_returns: pd.DataFrame, target_vol: float = 0.12,
                      periods: int = 252, max_leverage: float | None = None) -> float:
    """new_requirement（新增能力）：只返回缩放因子 scale = target_vol/port_vol
    （受 max_leverage 约束），便于在自定义权重基础上复用目标波动缩放逻辑。
    """
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns 必须是 DataFrame（每列一个资产）")
    if not (target_vol > 0):
        raise ValueError("target_vol 必须为正")
    if periods < 1:
        raise ValueError("periods 必须为正整数")
    if max_leverage is not None and max_leverage <= 0:
        raise ValueError("max_leverage 必须为正")
    r = asset_returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if r.shape[1] == 0 or r.shape[0] < 2:
        raise ValueError("asset_returns 至少需要 2 行有限收益与一个资产列")
    w0 = np.full(r.shape[1], 1.0 / r.shape[1])
    port_vol = _port_annual_vol(r, w0, periods)
    if not np.isfinite(port_vol) or port_vol <= 1e-12:
        return 0.0
    scale = target_vol / port_vol
    if max_leverage is not None:
        scale = min(scale, float(max_leverage))
    return float(scale)
