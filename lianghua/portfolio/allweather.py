"""全天候（风险预算）资产配置样例。

提供两类接口：
- all_weather_weights：有资产收益数据时用风险平价，否则用经典 30/40/15/15 代理配置。
- all_weather_portfolio：对资产日收益矩阵做配置并累积组合权益。
"""
from __future__ import annotations

import pandas as pd

from .optimizer import optimize, portfolio_returns

# 经典风险平价代理配置（股票 / 长期债券 / 商品 / 现金）
DEFAULT_ALLOC = {"equity": 0.30, "bond": 0.40, "commodity": 0.15, "cash": 0.15}


def all_weather_weights(asset_returns: pd.DataFrame | None = None,
                        method: str = "inverse_vol") -> pd.Series:
    """返回资产权重。有收益数据用逆波动（对零波动现金稳健），否则用默认配置。"""
    if asset_returns is not None and not asset_returns.dropna(how="all").empty:
        try:
            return optimize(asset_returns, method)
        except Exception:
            pass
    return pd.Series(DEFAULT_ALLOC)


def all_weather_portfolio(asset_returns: pd.DataFrame,
                           method: str = "risk_parity",
                           init_cash: float = 1_000_000.0) -> pd.Series:
    """对资产日收益矩阵做配置并累积组合权益。"""
    w = all_weather_weights(asset_returns, method)
    pr = portfolio_returns(w, asset_returns)
    return init_cash * (1.0 + pr).cumprod()
