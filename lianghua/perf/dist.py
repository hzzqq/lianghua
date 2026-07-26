"""收益分布特征：偏度 / 峰度 / 尾部比 / 历史 VaR。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _returns(equity: pd.Series) -> pd.Series:
    return equity.astype(float).pct_change().dropna()


def skew(equity: pd.Series) -> float:
    r = _returns(equity)
    return float(r.skew()) if len(r) > 2 else 0.0


def kurtosis(equity: pd.Series) -> float:
    r = _returns(equity)
    return float(r.kurt()) if len(r) > 3 else 0.0


def tail_ratio(equity: pd.Series, quantile: float = 0.05) -> float:
    """右尾(95%)绝对值 / 左尾(5%)绝对值。>1 右尾更厚(利好)。"""
    r = _returns(equity)
    if r.empty:
        return 0.0
    up = abs(r.quantile(1 - quantile))
    dn = abs(r.quantile(quantile))
    # 左尾为 0 时用 1e-12 兜底，避免向上游泄漏 inf（与全库其它比率处理一致）
    return float(up / (dn + 1e-12))


def value_at_risk(equity: pd.Series, alpha: float = 0.95) -> float:
    """历史法单期 VaR（正值为损失比例）。"""
    r = _returns(equity)
    if r.empty:
        return 0.0
    return float(-r.quantile(alpha))
