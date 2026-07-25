"""因子自动搜索 CLI：对某标的自动挖掘有效因子并输出 Top N + 单因子回测。

示例：
    python examples/run_factor.py --symbol 600519.SH --top 5
    python examples/run_factor.py --symbol 600519.SH --asset stock --forward 10 --out factors.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lianghua.data.gateway import DataGateway
from lianghua.core.assets import AssetType, detect_asset_type
from lianghua.factor import FactorEngine, FactorSearcher


def main():
    parser = argparse.ArgumentParser(description="Lianghua 自动因子挖掘 (RD-Agent 风格)")
    parser.add_argument("--symbol", required=True, help="标的代码，如 600519.SH")
    parser.add_argument("--asset", default=None, help="资产类别: stock/fund/future/option")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--top", type=int, default=5, help="返回因子数")
    parser.add_argument("--forward", type=int, default=5, help="持有期(交易日)")
    parser.add_argument("--out", default=None, help="可选：导出 Top 因子净值 CSV")
    args = parser.parse_args()

    at = AssetType(args.asset) if args.asset else detect_asset_type(args.symbol)
    df = DataGateway().fetch(args.symbol, args.start, args.end, asset=at)
    fs = FactorSearcher(df)
    top = fs.search(top_n=args.top, forward=args.forward)

    print(f"标的 {args.symbol} ({at.value}) 自动因子挖掘 Top{len(top)} (持有 {args.forward} 日):")
    for i, r in enumerate(top, 1):
        print(f"  {i}. {r['factor']:16} IC={r['ic']:+.4f}  分层多空={r['long_short_return']*100:+.2f}%  n={r['n']}")

    if args.out:
        fe = FactorEngine(df)
        eqs = {r["factor"]: fe.backtest_factor(fe.compute(r["factor"])) for r in top}
        pd.DataFrame(eqs).to_csv(args.out)
        print(f"因子净值已导出: {args.out}")


if __name__ == "__main__":
    main()
