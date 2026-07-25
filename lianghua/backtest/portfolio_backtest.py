"""多标的 / 多策略组合层回测。

对若干 (标的, 策略, 资产, 权重) 各自回测，按权重合成组合权益曲线，并输出子策略归因。
"""
from __future__ import annotations

import pandas as pd

from .runner import run_backtest
from ..perf.metrics import attribution


def portfolio_backtest(
    specs: list[dict],
    start: str = "2023-01-01",
    end: str = "2024-12-31",
    init_cash: float = 1_000_000.0,
) -> dict:
    """specs: [{"symbol","strategy","asset"(可选),"weight"(可选)}]。

    返回 dict: equity(组合权益序列), components(子策略权益矩阵),
    weights(权重 Series), attribution(收益归因 DataFrame)。
    """
    eqs: dict[str, pd.Series] = {}
    for spec in specs:
        sym = spec["symbol"]
        r = run_backtest(
            sym, start, end,
            strategy=spec.get("strategy", "sma_cross"),
            asset=spec.get("asset"),
        )
        eqs[sym] = r.equity
    comp = pd.DataFrame(eqs).ffill().dropna()
    w = pd.Series({s["symbol"]: float(s.get("weight", 1.0)) for s in specs})
    w = w.reindex(comp.columns).fillna(1.0)
    w = w / w.sum()
    norm = comp.div(comp.iloc[0])                 # 各子策略收益倍数
    equity = (norm * w).sum(axis=1) * init_cash
    attr = attribution({sym: comp[sym] for sym in comp.columns})
    return {
        "equity": equity,
        "components": comp,
        "weights": w,
        "attribution": attr,
    }
