"""真实网络数据 · 全资产实盘级回测示例。

对四类资产各取一个代表标的，走 DataGateway 真实源（AKShare/BaoStock），
断网/超时则自动降级演示数据（脚本会标注数据来源）。
结果打印成表，并输出缓存落盘情况。

用法：
    python examples/run_live_backtest.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lianghua.backtest.runner import run_backtest
from lianghua.core.assets import AssetType, detect_asset_type
from lianghua.data.gateway import DataGateway
from lianghua.perf.metrics import report

# 每类资产的代表标的（真实可拉取）
PLAN = [
    {"symbol": "600519.SH", "asset": "stock",  "strategy": "sma_cross",  "label": "贵州茅台(股票)"},
    {"symbol": "510300.SH", "asset": "fund",   "strategy": "momentum",   "label": "沪深300ETF(基金)"},
    {"symbol": "RB0.SHF",   "asset": "future", "strategy": "breakout",   "label": "螺纹主连(期货)"},
    {"symbol": "510050C3000.SH", "asset": "option", "strategy": "option_call_trend", "label": "50ETF购3000(期权)"},
]

START, END = "2023-01-01", "2023-12-31"


def data_source(gw: DataGateway, symbol: str, asset: AssetType) -> str:
    """判断该标的实际命中的是真实源还是演示数据（读缓存 source 列）。"""
    row = gw._cache_get(symbol, START, END, asset)
    if row is not None and not row.empty and "source" in row.columns and row["source"].notna().any():
        src = str(row["source"].dropna().iloc[0])
        return {
            "akshare": "真实(AKShare)", "baostock": "真实(BaoStock)",
            "csv": "本地CSV", "demo": "演示数据(降级)",
        }.get(src, src)
    return "演示数据(降级)"


def main():
    gw = DataGateway()
    rows = []
    print(f"{'标的':<18}{'资产':<8}{'策略':<16}{'数据来源':<14}{'期末权益':>12}{'年化':>9}{'夏普':>7}{'回撤':>8}")
    print("-" * 92)
    for item in PLAN:
        sym, asset = item["symbol"], AssetType(item["asset"])
        t0 = time.time()
        try:
            res = run_backtest(sym, START, END, strategy=item["strategy"],
                               asset=item["asset"], init_cash=1_000_000, gw=gw)
            src = data_source(gw, sym, asset)
            eq = res.equity
            m = report(eq, res.trades, df=res.df, init_cash=1_000_000)
            rows.append({
                "label": item["label"], "symbol": sym, "asset": item["asset"],
                "strategy": item["strategy"], "src": src,
                "final_equity": eq.iloc[-1], "annual": m.get("annual_return", 0),
                "sharpe": m.get("sharpe", 0), "mdd": m.get("max_drawdown", 0),
                "trades": len(res.trades),
            })
            print(f"{item['label']:<16}{item['asset']:<8}{item['strategy']:<16}{src:<14}"
                  f"{eq.iloc[-1]:>12,.0f}{m.get('annual_return',0)*100:>8.1f}%"
                  f"{m.get('sharpe',0):>7.2f}{m.get('max_drawdown',0)*100:>7.1f}%")
        except Exception as e:  # noqa
            print(f"{item['label']:<16} 回测异常: {repr(e)[:60]}")
        _ = time.time() - t0

    df = pd.DataFrame(rows)
    out = ROOT / "examples" / "live_backtest_result.csv"
    df.to_csv(out, index=False)
    print("-" * 92)
    print(f"完成 {len(rows)} 个资产回测，结果已写入 {out}")
    print("说明：数据网关已对命中的真实行情落 SQLite 缓存(data_cache.db)，后续离线可直接复用。")


if __name__ == "__main__":
    main()
