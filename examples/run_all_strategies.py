"""示例：遍历统一策略注册表中的全部单标的策略做回测并对比。

对单个标的，分别用每个已注册策略（迭代31-100 共 20 个）生成信号并向量化回测，
打印累计收益 / 夏普 / 最大回撤对比表。数据走 DataGateway 真实源，断网自动降级演示。

用法：
    python examples/run_all_strategies.py
    python examples/run_all_strategies.py 600519.SH stock
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lianghua.data.gateway import DataGateway
from lianghua.strategy.registry import STRATEGY_NAMES, get_strategy
from lianghua.backtest.vectorized import vectorized_backtest
from lianghua.core.assets import AssetType


def main(symbol: str = "600519.SH", asset: str = "stock",
         start: str = "2024-01-01", end: str = "2024-06-30"):
    gw = DataGateway()
    at = AssetType(asset)
    df = gw.fetch(symbol, start, end, asset=at)
    print(f"标的 {symbol}（{asset}）  数据行数={len(df)}  "
          f"来源={df['source'].iloc[0] if 'source' in df else 'NA'}\n")
    rows = []
    for name in STRATEGY_NAMES:
        try:
            sig = get_strategy(name).generate_signals(df)
            r = vectorized_backtest(df, sig)
            eq = r["equity"]
            tot = eq.iloc[-1] / eq.iloc[0] - 1
            rows.append({
                "策略": name,
                "累计%": round(tot * 100, 1),
                "夏普": round(r["metrics"]["sharpe"], 2),
                "回撤%": round(r["metrics"]["max_drawdown"] * 100, 1),
                "换手": r["num_changes"],
            })
        except Exception as e:
            rows.append({"策略": name, "累计%": None, "夏普": None,
                         "回撤%": None, "换手": f"ERR:{e}"[:20]})
    out = pd.DataFrame(rows).sort_values("累计%", ascending=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "600519.SH"
    ast = sys.argv[2] if len(sys.argv) > 2 else "stock"
    main(sym, ast)
