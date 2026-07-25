"""命令行回测入口（多资产）。

示例：
    # 股票
    python examples/run_backtest.py --symbol 600519.SH --strategy sma_cross
    # 期货
    python examples/run_backtest.py --symbol RB2410.SHF --asset future --strategy breakout
    # 期权
    python examples/run_backtest.py --symbol 510050C3000.SH --asset option --strategy option_call_trend
    # 基金定投
    python examples/run_backtest.py --symbol 110011.OF --asset fund --strategy dca
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lianghua.backtest.runner import run_backtest, SUPPORTED
from lianghua.perf.metrics import report


def main():
    ap = argparse.ArgumentParser("Lianghua Quant 多资产回测")
    ap.add_argument("--symbol", default="600519.SH")
    ap.add_argument("--asset", default=None, choices=["stock", "fund", "future", "option"],
                    help="资产类别（默认按代码自动识别）")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--strategy", default="sma_cross")
    ap.add_argument("--cash", type=float, default=1_000_000)
    ap.add_argument("--stop_loss", type=float, default=0.10)
    ap.add_argument("--out", default=None, help="结果 CSV 输出路径")
    args = ap.parse_args()

    asset = args.asset or ""
    supported = SUPPORTED.get(asset or "stock", [])
    if args.strategy not in supported:
        print(f"[警告] 策略 {args.strategy} 不在 {asset or 'auto'} 支持列表 {supported}，将尝试运行。")

    res = run_backtest(
        args.symbol, args.start, args.end, strategy=args.strategy,
        asset=args.asset, init_cash=args.cash, stop_loss=args.stop_loss,
    )
    perf = report(res.equity, res.trades, df=res.df, init_cash=args.cash)

    print(f"标的: {args.symbol}   资产: {args.asset or 'auto'}   策略: {args.strategy}")
    print(f"区间: {args.start} ~ {args.end}")
    print("-" * 52)
    labels = {
        "final_equity": "期末权益", "total_return": "总收益", "annual_return": "年化收益",
        "sharpe": "夏普比率", "max_drawdown": "最大回撤", "num_trades": "交易次数",
        "win_rate": "胜率", "benchmark_return": "基准收益", "excess_return": "超额收益",
    }
    for k, v in perf.items():
        if isinstance(v, float):
            if "return" in k or "rate" in k or "drawdown" in k:
                print(f"  {labels.get(k, k):>10}: {v * 100:+.2f}%")
            else:
                print(f"  {labels.get(k, k):>10}: {v:.2f}")
        else:
            print(f"  {labels.get(k, k):>10}: {v}")
    if args.out:
        res.to_frame().to_csv(args.out, index=False)
        print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
