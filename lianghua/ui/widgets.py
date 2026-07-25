"""UI 共用图表组件（复用红涨绿跌配色）。"""
from __future__ import annotations

import pandas as pd

# A 股配色：涨红 / 跌绿
UP = "#ff4d4f"
DOWN = "#00d486"
ACCENT = "#667eea"


def equity_fig(equity: pd.Series, title: str = "净值曲线"):
    """返回 plotly 净值曲线 Figure（暗色、红涨绿跌）。"""
    import plotly.graph_objects as go
    eq = equity.astype(float)
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


def drawdown_fig(equity: pd.Series, title: str = "回撤"):
    import plotly.graph_objects as go
    eq = equity.astype(float)
    dd = (eq / eq.cummax() - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index.astype(str).tolist(), y=dd.values,
        fill="tozeroy", line=dict(color=DOWN, width=1), name="回撤%",
    ))
    fig.update_layout(template="plotly_dark", title=title, height=260)
    return fig
