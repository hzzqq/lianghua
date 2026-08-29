# -*- coding: utf-8 -*-
"""量化小白向导：一条命令跑完「取数 → 选策略 → 回测 → 可信度体检」。

面向完全没用过量化的用户：不需要懂参数、不需要写代码，跟着提示选就行。

三种用法
--------
1) 全程向导（推荐第一次用）::

    python tools/quickstart.py

2) 指定标的和策略（跳过提问）::

    python tools/quickstart.py --symbol 600519.SH --strategy sma_cross

3) 完全零参数体验（用内置演示数据，断网也能跑）::

    python tools/quickstart.py --demo

输出会告诉你三件事：赚了多少、这个结果**可不可信**、以及下一步该看什么。
可信度体检是本框架的关键能力——回测跑通不等于结果可信。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lianghua.backtest.engine import BacktestEngine  # noqa: E402
from lianghua.backtest.validate import out_of_sample_report  # noqa: E402
from lianghua.strategy.registry import STRATEGY_NAMES, get_strategy  # noqa: E402

# 给小白的推荐策略：名称 -> 一句话说明（挑最易理解的 6 个）
RECOMMENDED = [
    ("sma_cross", "双均线：短期均线上穿长期买入，下穿卖出。最经典，适合入门"),
    ("rsi_strategy", "RSI 超买超卖：涨太多就卖，跌太多就买"),
    ("bollinger", "布林带：价格碰上轨卖、碰下轨买，震荡市好用"),
    ("macd", "MACD：判断趋势强弱与转折，趋势市常用"),
    ("momentum", "动量：买最近涨得好的，追涨"),
    ("mean_reversion", "均值回归：跌多了买、涨多了卖，与动量相反"),
]
_RECOMMENDED_NAMES = {n for n, _ in RECOMMENDED if n in STRATEGY_NAMES}

DEMO_DAYS = 500


def _line(ch: str = "=", n: int = 68) -> str:
    return ch * n


def _ask_choice(prompt: str, options: list, default: int = 0) -> int:
    """带编号的选择题，直接回车用默认值；输入非法也回退默认值。"""
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    try:
        raw = input(f"{prompt} [默认 {default + 1}] ").strip()
    except (EOFError, KeyboardInterrupt):
        # 输入被关闭/中断（管道、CI、用户 Ctrl-C）：直接用默认值，不让向导崩溃
        print()
        return default
    if not raw:
        return default
    try:
        idx = int(raw) - 1
    except ValueError:
        return default
    return idx if 0 <= idx < len(options) else default


def _demo_data(days: int = DEMO_DAYS) -> pd.DataFrame:
    """内置演示数据：确定性几何随机游走，断网也能完整体验流程。"""
    rng = np.random.default_rng(20260829)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.018, days))
    noise = np.abs(rng.normal(0, np.std(close) * 0.3 + 0.01, days))
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    return pd.DataFrame(
        {"open": close * (1 + rng.normal(0, 0.002, days)),
         "high": close + noise, "low": close - noise,
         "close": close, "volume": np.abs(rng.normal(1e6, 3e5, days)) + 1e4},
        index=idx,
    )


def _fetch(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """取真实数据，并如实告知来源。

    数据网关在真实源不可用时会自动降级为演示数据。这里必须把这件事说清楚：
    拿假数据跑出来的漂亮收益最容易误导人，所以来源与降级状态一律如实标注。
    """
    try:
        from lianghua.data.gateway import DataGateway  # noqa: PLC0415
        gw = DataGateway()
        df = gw.fetch(symbol, start, end, timeout=8.0)
        if df is not None and len(df) > 0:
            was_demo = bool(getattr(gw, "last_was_demo", False))
            src = str(getattr(gw, "last_source", "unknown"))
            if was_demo:
                print(f"  真实源不可用，已降级为演示数据（来源={src}），结果不是真实行情")
                return df, f"演示数据（真实源不可用：{src}，非真实行情）"
            return df, f"真实行情 {symbol}（来源 {src}）"
    except Exception as e:  # noqa: BLE001
        print(f"  取真实数据失败（{type(e).__name__}: {e}）")
    print("  已回退到内置演示数据（随机生成，仅用于体验流程，不是真实行情）")
    return _demo_data(), "演示数据（非真实行情）"


def _fmt_pct(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{v:+.2%}" if np.isfinite(v) else "-"


def _print_sanity(sn: dict):
    """可信度体检输出：把「结果能不能信」翻译成人话。"""
    print()
    print(_line("-"))
    print("【可信度体检】—— 这一步比收益数字更重要")
    print(_line("-"))
    if sn["level"] == "ok":
        print("  通过：未发现明显的可信度问题。")
        return
    icon = {"warn": "提醒", "error": "警告"}.get(sn["level"], "提醒")
    for it in sn["issues"]:
        print(f"  [{icon}] {it['msg']}")
    if sn["level"] == "error":
        print("\n  存在致命问题，该结果不可用，请检查数据与参数。")
    else:
        print("\n  以上为提醒项，结果可参考，但别把回测收益直接当成实盘预期。")


def _print_oos(rep: dict):
    print()
    print(_line("-"))
    print("【样本外验证】—— 检验是不是「拟合出来的好结果」")
    print(_line("-"))
    if not rep.get("ok"):
        print(f"  {rep.get('note', '无法评估')}")
        return
    si, so = rep["in_sample"], rep["out_sample"]
    print(f"  样本内（前 60%，用于观察）：总收益 {_fmt_pct(si['total_return'])}，"
          f"交易 {si['n_trades']} 笔")
    print(f"  样本外（后 40%，真正考验）：总收益 {_fmt_pct(so['total_return'])}，"
          f"交易 {so['n_trades']} 笔")
    label = {"robust": "稳健", "degraded": "减弱", "overfit": "疑似过拟合",
             "no_edge": "无明显优势"}.get(rep["verdict"], rep["verdict"])
    print(f"  判定：{label} —— {rep['note']}")


def _next_steps(res, sn: dict, rep: dict):
    print()
    print(_line("-"))
    print("【下一步建议】")
    print(_line("-"))
    tips = []
    if sn["level"] != "ok":
        tips.append("先处理上面的可信度提醒，再看收益数字")
    if rep.get("ok") and rep.get("verdict") == "overfit":
        tips.append("换更简单的策略或减少参数，过拟合的策略实盘通常失效")
    if not tips:
        tips.append("换 2~3 个策略横向对比，别只看一个结果")
    tips.append("用更长的时间区间再跑一次（建议 >=2 年）")
    tips.append(f"在终端里看图形化结果：运行 python start.py 后打开『策略回测』页")
    for i, t in enumerate(tips, 1):
        print(f"  {i}. {t}")


def run_once(symbol: str | None, strategy: str, start: str, end: str,
             demo: bool, init_cash: float) -> int:
    print(_line())
    print("量化小白向导")
    print(_line())

    # 1) 数据
    if demo or not symbol:
        df, source = _demo_data(), "演示数据（非真实行情）"
    else:
        df, source = _fetch(symbol, start, end)
    print(f"\n数据：{source}，共 {len(df)} 根 K 线")

    # 2) 策略
    if strategy not in STRATEGY_NAMES:
        print(f"  未找到策略 {strategy!r}，回退到 sma_cross")
        strategy = "sma_cross"
    print(f"策略：{strategy}")

    # 3) 回测（默认参数即无前视：信号延迟一期、次日开盘成交）
    try:
        signals = get_strategy(strategy).generate_signals(df)
        res = BacktestEngine(init_cash=init_cash).run(df, signals)
    except Exception as e:  # noqa: BLE001
        print(f"\n回测失败：{type(e).__name__}: {e}")
        return 1

    st = res.stats()
    print()
    print(_line())
    print("【回测结果】")
    print(_line())
    print(f"  总收益     {_fmt_pct(st['total_return'])}")
    print(f"  最大回撤   {_fmt_pct(st['max_drawdown'])}")
    print(f"  期末资金   {st['final_equity']:,.2f}（起始 {init_cash:,.0f}）")
    print(f"  交易次数   {st['n_trades']}")
    fill_cn = {"open": "开盘", "close": "收盘"}.get(st["fill_price"], st["fill_price"])
    print(f"  执行假设   信号延迟 {st['execution_lag']} 期、以{fill_cn}价成交"
          f"（无前视偏差）")

    sn = res.sanity()
    rep = out_of_sample_report(df, get_strategy(strategy).generate_signals)
    _print_sanity(sn)
    _print_oos(rep)
    _next_steps(res, sn, rep)
    print()
    print(_line())
    print("提示：以上均为历史回测，不构成投资建议。")
    print(_line())
    return 0


def interactive() -> tuple[str | None, str, bool]:
    print(_line())
    print("量化小白向导 —— 跟着提示选就行，直接回车表示用默认值")
    print(_line())

    print("\n第 1 步：用真实数据还是先体验？")
    idx = _ask_choice("请选择", [
        "内置演示数据（不用联网，第一次用推荐这个）",
        "真实 A 股行情（需要联网，断网会自动回退演示数据）",
    ])
    if idx == 0:
        print("\n第 2 步：选策略")
        opts = [f"{n} —— {d}" for n, d in RECOMMENDED if n in _RECOMMENDED_NAMES]
        sy = _ask_choice("请选择", opts)
        names = [n for n, _ in RECOMMENDED if n in _RECOMMENDED_NAMES]
        return None, (names[sy] if names else "sma_cross"), True

    try:
        symbol = input("\n请输入股票代码（如 600519.SH）：").strip() or "600519.SH"
    except (EOFError, KeyboardInterrupt):
        print()
        symbol = "600519.SH"
    print("\n第 2 步：选策略")
    opts = [f"{n} —— {d}" for n, d in RECOMMENDED if n in _RECOMMENDED_NAMES]
    sy = _ask_choice("请选择", opts)
    names = [n for n, _ in RECOMMENDED if n in _RECOMMENDED_NAMES]
    return symbol, (names[sy] if names else "sma_cross"), False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="量化小白向导：一条命令跑完回测并给出可信度体检")
    ap.add_argument("--symbol", help="标的代码，如 600519.SH")
    ap.add_argument("--strategy", default="sma_cross", help=f"策略名，可选：{STRATEGY_NAMES}")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金，默认 100 万")
    ap.add_argument("--demo", action="store_true", help="用内置演示数据，断网也能跑")
    args = ap.parse_args(argv)

    if args.symbol or args.demo:
        return run_once(args.symbol, args.strategy, args.start, args.end,
                        args.demo, args.cash)

    if not sys.stdin.isatty():
        # 非交互环境（CI/管道）：没有可问的对象，直接用演示数据跑一遍
        print("检测到非交互环境，自动使用演示数据。")
        return run_once(None, args.strategy, args.start, args.end, True, args.cash)

    symbol, strategy, demo = interactive()
    return run_once(symbol, strategy, args.start, args.end, demo, args.cash)


if __name__ == "__main__":
    raise SystemExit(main())
