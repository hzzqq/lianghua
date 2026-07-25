"""迭代 100 收尾：统一接入骨架集成测试。

验证：
- 策略统一注册表（20 个单标的策略）全部可生成信号
- 优化器注册表（17 个）全部可产出权重
- orchestrator.run_plan 支持 optimizer 选权 + report 出 tearsheet
- 能力索引 list_capabilities / summary_counts 正常
全离线（桩网关注入合成行情），不依赖网络。
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from lianghua.strategy.registry import get_strategy, STRATEGY_NAMES
from lianghua.portfolio.registry import get_optimizer, OPTIMIZER_NAMES
from lianghua.backtest.orchestrator import run_plan
from lianghua.core.capabilities import list_capabilities, summary_counts


def _demo_df(n=300, seed=1):
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), index=dates)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 1e5,
    })


class StubGW:
    """离线确定性桩网关：按 symbol 后缀返回合成行情。"""
    def fetch(self, symbol, start, end, asset=None):
        n = 300
        return _demo_df(n=n, seed=hash(symbol) % 1000)


def ok(label, cond):
    print(("  ✓ " if cond else "  ✗ ") + label)
    if not cond:
        raise AssertionError(label + " 校验失败")


def test_strategy_registry():
    print("[1] 策略统一注册表")
    ok("策略数量>=20", len(STRATEGY_NAMES) >= 20)
    df = _demo_df()
    for name in STRATEGY_NAMES:
        sig = get_strategy(name).generate_signals(df)
        ok(f"策略 {name} 信号有效", isinstance(sig, pd.Series) and len(sig) == len(df))


def test_optimizer_registry():
    print("[2] 优化器注册表")
    ok("优化器数量>=15", len(OPTIMIZER_NAMES) >= 15)
    np.random.seed(0)
    A = pd.DataFrame(np.random.randn(250, 5) * 0.02,
                    columns=list("ABCDE"))
    for name in OPTIMIZER_NAMES:
        w = get_optimizer(name)(A)
        ok(f"优化器 {name} 权重有效",
           isinstance(w, pd.Series) and len(w) == 5 and np.all(np.isfinite(w.to_numpy(dtype=float))))


def test_orchestrator():
    print("[3] orchestrator.run_plan（含优化器/报告）")
    plan = [
        {"symbol": "600519.SH", "asset": "stock", "strategy": "sma_cross"},
        {"symbol": "510050.SH", "asset": "stock", "strategy": "rsi"},
        {"symbol": "000001.SZ", "asset": "stock", "strategy": "bollinger"},
    ]
    gw = StubGW()
    # 默认权重
    r0 = run_plan(plan, gw=gw)
    ok("默认组合净值有效", isinstance(r0["equity"], pd.Series) and len(r0["equity"]) == 300)
    ok("默认权重和≈1", abs(sum(r0["weights"].values()) - 1) < 1e-3)
    # 用优化器选权
    r1 = run_plan(plan, gw=gw, optimizer="risk_parity")
    ok("优化器被采纳", r1["optimizer"] == "risk_parity")
    ok("优化器权重和≈1", abs(sum(r1["weights"].values()) - 1) < 1e-3)
    ok("优化器组合净值有效", isinstance(r1["equity"], pd.Series) and len(r1["equity"]) > 0)
    # 报告
    r2 = run_plan(plan, gw=gw, optimizer="min_variance", report=True)
    ok("tearsheet 报告存在", "report" in r2 and isinstance(r2["report"], pd.DataFrame))
    # 退化优化器回退（强制一个会出全零权重的极端场景由 orchestrator 兜底）
    # 用不存在的策略应报错
    try:
        run_plan([{"symbol": "X", "strategy": "nope"}], gw=gw)
        ok("未知策略报错", False)
    except ValueError:
        ok("未知策略报错", True)


def test_capabilities():
    print("[4] 能力索引")
    cap = list_capabilities()
    ok("策略清单非空", len(cap["strategies"]) >= 20)
    ok("优化器清单非空", len(cap["optimizers"]) >= 15)
    ok("期权策略清单非空", len(cap["option_strategies"]) >= 1)
    ok("风险模块非空", len(cap["risk"]) >= 10)
    ok("绩效模块非空", len(cap["perf"]) >= 10)
    sc = summary_counts()
    ok("计数合理", sc["strategies"] >= 20 and sc["optimizers"] >= 15)


if __name__ == "__main__":
    test_strategy_registry()
    test_optimizer_registry()
    test_orchestrator()
    test_capabilities()
    print("\n=== 集成测试全部通过 ===")
