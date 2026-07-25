"""UI 共用图表组件（复用红涨绿跌配色）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A 股配色：涨红 / 跌绿
UP = "#ff4d4f"
DOWN = "#00d486"
ACCENT = "#667eea"


def _finite(v) -> bool:
    """判断是否为有限实数。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")


def safe_pct(v, nd: int = 1) -> str:
    """把比值格式为百分比；非有限值返回 N/A（避免 UI 出现 nan%）。"""
    return f"{v * 100:.{nd}f}%" if _finite(v) else "N/A"


def safe_num(v, nd: int = 2) -> str:
    """把数值格式为定长小数；非有限值返回 N/A。"""
    return f"{v:.{nd}f}" if _finite(v) else "N/A"


def _finite_series(s: "pd.Series") -> "pd.Series":
    """清洗为有限浮点序列（剔除 NaN/±inf），空序列返回空 Series。"""
    return pd.Series(s).astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def empty_figure(message: str, height: int = 300) -> "object":
    """返回一个带占位提示的空 Figure（用于空态 / 错误态降级）。"""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper",
        x=0.5, y=0.5, font=dict(color="#9aa0c0", size=14),
    )
    fig.update_layout(
        height=height, template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def equity_fig(equity: "pd.Series", title: str = "净值曲线"):
    """返回 plotly 净值曲线 Figure（暗色、红涨绿跌）。空/全非有限时降级占位。"""
    import plotly.graph_objects as go

    eq = _finite_series(equity)
    if eq.empty:
        return empty_figure("无有效净值数据", height=360)
    color = UP if eq.iloc[-1] >= eq.iloc[0] else DOWN
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index.astype(str).tolist(), y=eq.values,
        mode="lines", line=dict(color=color, width=2), name="净值",
    ))
    fig.update_layout(
        template="plotly_dark", title=title, height=360,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def drawdown_fig(equity: "pd.Series", title: str = "回撤"):
    """返回 plotly 回撤面积图。空/全非有限时降级占位。"""
    import plotly.graph_objects as go

    eq = _finite_series(equity)
    if eq.empty:
        return empty_figure("无有效净值数据", height=260)
    dd = (eq / eq.cummax() - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index.astype(str).tolist(), y=dd.values,
        fill="tozeroy", line=dict(color=DOWN, width=1), name="回撤%",
    ))
    fig.update_layout(template="plotly_dark", title=title, height=260)
    return fig
