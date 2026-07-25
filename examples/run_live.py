"""实盘/纸面交易控制台。

用法：
  python examples/run_live.py --broker paper --strategy sma_cross \\
      --symbols 600519.SH,000300.SH --capital 1e6 --lookback 120 --cycles 5

broker 取值：sim / paper / qmt / pt
  - paper ：真实行情网关 + 模拟成交，零资金风险（默认，用于预演连接实盘）
  - qmt/pt：真实柜台，需 SDK + 客户端 + 账户；未就绪会明确报错
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

from lianghua.execution.brokers import make_broker
from lianghua.execution.live import LiveEngine
from lianghua.data.gateway import DataGateway


def main():
    ap = argparse.ArgumentParser(description="Lianghua 实盘/纸面交易控制台")
    ap.add_argument("--broker", default="paper", choices=["sim", "paper", "qmt", "pt"])
    ap.add_argument("--strategy", default="sma_cross")
    ap.add_argument("--symbols", default="600519.SH")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--lookback", type=int, default=120)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--account-id", default="")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="每次循环间隔秒（0=立即连续跑完）")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    broker_cfg = {}
    if args.broker in ("qmt", "pt") and args.account_id:
        broker_cfg["account_id"] = args.account_id

    broker = make_broker(args.broker, init_cash=args.capital, **broker_cfg)
    gw = DataGateway()
    engine = LiveEngine(
        broker, gw, args.strategy, symbols,
        capital=args.capital, lookback=args.lookback,
        mode=("live" if args.broker in ("qmt", "pt") else "paper"),
        live=args.broker in ("qmt", "pt"),
    )
    print(f"[live] broker={args.broker} strategy={args.strategy} "
          f"symbols={symbols} mode={engine.mode} live={engine.live}")
    for i in range(args.cycles):
        try:
            step = engine.step()
            acts = step.get("actions", [])
            print(f"[cycle {i+1}] 动作数={len(acts)}")
            for a in acts:
                print("   ", a)
        except Exception as e:
            print(f"[cycle {i+1}] ERROR: {e}")
            break
        if args.interval > 0:
            time.sleep(args.interval)
    snap = engine.status()
    print("[snapshot] account:", snap.get("account"))


if __name__ == "__main__":
    main()
