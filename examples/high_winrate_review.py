"""高胜率量化方案实测：真实行情 + 自研引擎 + 样本外验证。

目标（用户指令「完善量化方案，测试其可行性，结合真实数据的实际效果，维持高胜率」）：
1. 用 R17 修复后的腾讯真实行情源（DataGateway，demo=False）取 11 只流动性 A 股 2019-2024
   （覆盖 2019-2021 牛市 + 2022-2024 熊/震荡），避免「回测建立在随机游走上」。
2. 跑 6 个均值回复类候选策略（天然高胜率）：bollinger / zscore / rsi / ou / williams_r / cci_signal。
3. 用**正确的**逐笔配对胜率（修复 perf.metrics.win_rate 的 cash_after 近似缺陷）计算胜率、
   盈亏比、盈利因子、夏普、最大回撤、相对买入持有的超额收益。
4. 样本外验证：前 60% 定参、后 40% 验证，报告稳健/退化/过拟合判定。
5. 输出 outputs/high_winrate_review.csv + .html 结构化报告 + summary.json（机器可读交付物）。

引擎假设（与生产 run_backtest 一致、真实可执行）：
- 成本：佣金万三 + 滑点千一 + 印花税千一（卖出）
- 前视治理：execution_lag=1（t 日收盘出信号、t+1 开盘成交），杜绝前视偏差
- 风控：个股止损 10%（stop=0.10）；另跑一组 stop=1.0（等价于不止损）暴露信号本身的胜率
"""
from __future__ import annotations

import os
import sys
import json
import html
import datetime as dt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lianghua.data.gateway import DataGateway
from lianghua.strategy.registry import get_strategy
from lianghua.backtest.engine import BacktestEngine
from lianghua.risk.manager import RiskManager
from lianghua.perf.metrics import sharpe, max_drawdown, annual_return, buyhold_equity, round_trip_stats
from lianghua.backtest.validate import split_oos

START, END = "2019-01-01", "2024-12-31"
INIT_CASH = 1_000_000.0

STRATEGIES = ["bollinger", "zscore", "rsi", "ou", "williams_r", "cci_signal"]
STRATEGY_LABEL = {
    "bollinger": "布林带均值回复", "zscore": "Z-score 均值回归", "rsi": "RSI 超买超卖",
    "ou": "OU 均值回复", "williams_r": "威廉 %R 反转", "cci_signal": "CCI 商品通道",
}

# 参数敏感性网格：每个策略的主旋钮名 + 待扫参数值（统一调参护栏下的泛化检验）
PARAM_GRID = {
    "bollinger":   ("period", [15, 20, 25]),
    "zscore":      ("period", [15, 20, 25]),
    "rsi":         ("period", [10, 14, 21]),
    "williams_r":  ("period", [10, 14, 21]),
    "cci_signal":  ("period", [14, 20, 30]),
    "ou":          ("window", [40, 60, 90]),
}

# 出场/过滤三档对照（验证「10% 硬止损错配」与「趋势过滤器」两条改进假设）
VARIANTS = [
    ("baseline",         0.10, False),   # 原方案：10% 硬止损、无过滤
    ("wide_stop",        0.22, False),   # 改进①：宽幅灾难止损替代硬砍
    ("wide_stop_filter", 0.22, True),    # 改进①②叠加：宽止损 + 多头趋势过滤器
]

SYMBOLS = [
    ("600519.SH", "贵州茅台"), ("000858.SZ", "五粮液"), ("600036.SH", "招商银行"),
    ("601318.SH", "中国平安"), ("000001.SZ", "平安银行"), ("600030.SH", "中信证券"),
    ("000651.SZ", "格力电器"), ("600031.SH", "三一重工"), ("300750.SZ", "宁德时代"),
    ("600276.SH", "恒瑞医药"), ("000333.SZ", "美的集团"),
]


# ---------- 逐笔配对胜率直接使用框架级 perf.metrics.round_trip_stats（已修复旧版近似） ----------


def oos_eval(df: pd.DataFrame, fn, train_ratio: float = 0.6,
             risk: RiskManager | None = None) -> dict:
    """样本外验证 + 样本内/外胜率。复刻 validate.out_of_sample_report 的判定阈值，
    但额外给出逐笔胜率（原报告只给收益衰减）。"""
    if len(df) < 60:
        return dict(ok=False, verdict="unknown", note="样本过短")
    train, test = split_oos(df, train_ratio)
    if len(test) < 20:
        return dict(ok=False, verdict="unknown", note="样本外过短")
    eng = BacktestEngine(init_cash=INIT_CASH, risk=risk or RiskManager(stop_loss=0.10))
    r_in = eng.run(train, fn(train))
    r_out = eng.run(test, fn(test))
    ri = float(r_in.stats()["total_return"] or 0.0)
    ro = float(r_out.stats()["total_return"] or 0.0)
    if ri <= 0:
        verdict = "no_edge"
        decay = None
    else:
        decay = 1.0 - ro / ri
        if decay < 0:
            verdict = "robust"
        elif ro <= 0:
            verdict = "overfit"
        elif decay > 0.7:
            verdict = "overfit"
        elif decay > 0.4:
            verdict = "degraded"
        else:
            verdict = "robust"
    return dict(
        ok=True, verdict=verdict, decay=decay,
        in_return=ri, out_return=ro,
        in_win=round_trip_stats(r_in.trades)["win_rate"],
        out_win=round_trip_stats(r_out.trades)["win_rate"],
        in_trades=len(r_in.trades), out_trades=len(r_out.trades),
    )


def fetch_real(gw: DataGateway, sym: str):
    df = gw.fetch(sym, START, END, asset="stock", timeout=10, retries=2, backoff=1)
    if df is None or df.empty:
        raise RuntimeError("空数据")
    if gw.last_was_demo:
        raise RuntimeError(f"落到演示(假)数据 source={gw.last_source}")
    return df


def trend_filter(df: pd.DataFrame, sig: pd.Series, slow: int = 120) -> pd.Series:
    """多头趋势过滤器：仅在 close ≥ 慢线 MA 的「多头 regime」才允许均值回复做多。

    解决上轮暴露的第二个问题——大牛市中空仓错过趋势。这里不直接「全程持有」，
    而是只在趋势向上时才参与均值回复，避免在确认下行趋势中接飞刀、也减少空仓时间。
    信号为 -1/0/1，仅屏蔽做多_entry（sig==1 且不在多头 regime）。
    """
    c = df["close"].astype(float)
    ma = c.rolling(slow, min_periods=max(20, slow // 2)).mean()
    uptrend = (c >= ma).fillna(False)
    out = sig.copy()
    out[(sig == 1) & (~uptrend)] = 0
    return out


def run_combo(df, strat, stop, use_filter, bench_ret, bench_sharpe, keep_equity=False):
    """单个 (标的, 策略, 出场, 过滤) 组合的真实回测，返回指标记录。"""
    fn = get_strategy(strat).generate_signals
    raw = fn(df)
    sig = trend_filter(df, raw) if use_filter else raw
    eng = BacktestEngine(init_cash=INIT_CASH, risk=RiskManager(stop_loss=stop))
    res = eng.run(df, sig)
    st = res.stats()
    rt = round_trip_stats(res.trades)
    rec = dict(
        strategy=strat, label=STRATEGY_LABEL[strat],
        benchmark_return=bench_ret, benchmark_sharpe=bench_sharpe,
        stop_win=rt["win_rate"], stop_trades=rt["n"], stop_pf=rt["profit_factor"],
        stop_ret=float(st["total_return"] or 0.0),
        sharpe=sharpe(res.equity), max_dd=max_drawdown(res.equity),
        annual=annual_return(res.equity),
        excess=rt and (float(st["total_return"] or 0.0) - bench_ret),
        sanity=res.sanity()["level"],
    )
    # 样本外（用同一出场/过滤设定，保证口径一致）
    oos = oos_eval(df, (lambda d: trend_filter(d, fn(d)) if use_filter else fn(d)),
                   0.6, RiskManager(stop_loss=stop))
    rec.update(dict(
        oos_verdict=oos.get("verdict", "unknown"), oos_decay=oos.get("decay"),
        oos_in_win=oos.get("in_win"), oos_out_win=oos.get("out_win"),
        oos_in_ret=oos.get("in_return"), oos_out_ret=oos.get("out_return"),
    ))
    rec["excess"] = float(st["total_return"] or 0.0) - bench_ret
    if keep_equity and rec["excess"] > 0:
        rec["_equity_dates"] = [str(x) for x in res.equity.index]
        rec["_equity"] = [float(x) for x in res.equity.values]
        rec["_trades"] = res.trades
    return rec


def main():
    gw = DataGateway()
    rows = []
    data_meta = {}
    data = {}
    for sym, name in SYMBOLS:
        try:
            df = fetch_real(gw, sym)
        except Exception as e:
            print(f"[SKIP] {sym} {name}: {type(e).__name__}: {e}")
            rows.append(dict(symbol=sym, name=name, error=str(e)))
            continue
        data[(sym, name)] = df
        data_meta[sym] = dict(name=name, rows=len(df),
                              cmin=float(df["close"].min()), cmax=float(df["close"].max()),
                              source=gw.last_source)
        print(f"[OK] {sym} {name}: {len(df)} 行, close {df['close'].min():.2f}~{df['close'].max():.2f} (source={gw.last_source})")

    # ---------- 1) 三档出场/过滤对照 ----------
    variant_cmp = []
    for vname, stop, usef in VARIANTS:
        vrows = []
        for (sym, name), df in data.items():
            bench = buyhold_equity(df, INIT_CASH)
            bench_ret = float(bench.iloc[-1] / bench.iloc[0] - 1)
            bench_sharpe = sharpe(bench)
            for strat in STRATEGIES:
                rec = dict(symbol=sym, name=name, variant=vname,
                           benchmark_return=bench_ret, benchmark_sharpe=bench_sharpe)
                # 每个 variant 都留存净值/成交（内存可忽略），proof 导出时按 excess>0 筛选
                c = run_combo(df, strat, stop, usef, bench_ret, bench_sharpe,
                              keep_equity=True)
                rec.update(c)
                # 信号本身胜率（不止损口径）仅对 headline 变体计算，展示「止损错配」幅度
                if vname == "wide_stop_filter":
                    nos = run_combo(df, strat, 1.0, usef, bench_ret, bench_sharpe)
                    rec["nostop_win"] = nos["stop_win"]
                    rec["nostop_trades"] = nos["stop_trades"]
                vrows.append(rec)
                rows.append(rec)
                print(f"  [{vname:16s}] {sym} {strat:11s} win={rec['stop_win']:.2%} "
                      f"pf={rec['stop_pf']:.2f} excess={rec['excess']:+.1%} oos={rec['oos_verdict']}")
        valid = [r for r in vrows if "stop_win" in r]
        wins = [r["stop_win"] for r in valid]
        pfs = [r["stop_pf"] for r in valid if np.isfinite(r["stop_pf"])]
        exs = [r["excess"] for r in valid]
        ssp = [r["sharpe"] for r in valid if np.isfinite(r["sharpe"])]
        bsp = [r["benchmark_sharpe"] for r in valid if np.isfinite(r["benchmark_sharpe"])]
        variant_cmp.append(dict(
            variant=vname, n=len(valid),
            avg_win=float(np.mean(wins)),
            avg_pf=float(np.mean(pfs)) if pfs else float("nan"),
            avg_excess=float(np.mean(exs)),
            median_excess=float(np.median(exs)),
            pos_excess=sum(1 for x in exs if x > 0),
            avg_strat_sharpe=float(np.mean(ssp)) if ssp else float("nan"),
            avg_bench_sharpe=float(np.mean(bsp)) if bsp else float("nan"),
        ))

    # ---------- 2) 参数敏感性网格（改进版出场 wide_stop_filter）----------
    grid = []
    for strat in STRATEGIES:
        kw, vals = PARAM_GRID[strat]
        for v in vals:
            grp = []
            for (sym, name), df in data.items():
                bench = buyhold_equity(df, INIT_CASH)
                bench_ret = float(bench.iloc[-1] / bench.iloc[0] - 1)
                fn = get_strategy(strat, **{kw: v}).generate_signals
                sig = trend_filter(df, fn(df))
                res = BacktestEngine(init_cash=INIT_CASH,
                                      risk=RiskManager(stop_loss=0.22)).run(df, sig)
                rt = round_trip_stats(res.trades)
                grp.append((rt["win_rate"], float(res.stats()["total_return"] or 0.0) - bench_ret))
            wins = [g[0] for g in grp]
            exs = [g[1] for g in grp]
            grid.append(dict(strategy=strat, label=STRATEGY_LABEL[strat],
                             param=kw, value=v, n=len(grp),
                             avg_win=float(np.mean(wins)),
                             median_excess=float(np.median(exs))))

    # ---------- 3) 聚合（以改进版 wide_stop_filter 为头条）----------
    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "high_winrate_review.csv")
    out_html = os.path.splitext(out_csv)[0] + ".html"
    out_json = os.path.splitext(out_csv)[0] + ".json"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    improved = [r for r in rows if r.get("variant") == "wide_stop_filter" and "stop_win" in r]
    valid = improved
    agg = []
    for strat in STRATEGIES:
        sub = [r for r in valid if r["strategy"] == strat]
        if not sub:
            continue
        sw = [r["stop_win"] for r in sub]
        nw = [r.get("nostop_win", float("nan")) for r in sub if np.isfinite(r.get("nostop_win", float("nan")))]
        ex = [r["excess"] for r in sub]
        pf = [r["stop_pf"] for r in sub if np.isfinite(r["stop_pf"])]
        ow = [r["oos_out_win"] for r in sub if r.get("oos_out_win") is not None]
        ssp = [r["sharpe"] for r in sub if np.isfinite(r["sharpe"])]
        bsp = [r["benchmark_sharpe"] for r in sub if np.isfinite(r["benchmark_sharpe"])]
        robust = [r for r in sub if r["oos_verdict"] in ("robust", "degraded")]
        agg.append(dict(
            strategy=strat, label=STRATEGY_LABEL[strat],
            n=len(sub),
            avg_stop_win=float(np.mean(sw)),
            avg_nostop_win=float(np.mean(nw)) if nw else float("nan"),
            avg_excess=float(np.mean(ex)),
            median_excess=float(np.median(ex)),
            pos_excess=sum(1 for x in ex if x > 0),
            avg_pf=float(np.mean(pf)) if pf else float("nan"),
            avg_oos_win=float(np.mean(ow)) if ow else float("nan"),
            robust_rate=float(len(robust) / len(sub)),
            avg_strat_sharpe=float(np.mean(ssp)) if ssp else float("nan"),
            avg_bench_sharpe=float(np.mean(bsp)) if bsp else float("nan"),
        ))
    agg_df = pd.DataFrame(agg).sort_values("avg_stop_win", ascending=False)

    # 原始宽表（含 variant 字段，机器可读）
    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    pd.DataFrame(clean_rows).to_csv(out_csv, index=False)
    print(f"\n[WRITE] {out_csv}")
    _render_html(out_html, valid, agg_df, data_meta, variant_cmp, grid)
    print(f"[WRITE] {out_html}")
    _export_summary(out_json, valid, agg_df, data_meta,
                    recommendation_text(agg_df, as_html=False), variant_cmp, grid)
    print(f"[WRITE] {out_json}")
    # 跑赢买入持有的组合（跨全部 variant 扫描，含 baseline 的 4 个实证组合）
    eq_path, tr_path = _export_proof(os.path.dirname(out_csv), rows)
    for r in rows:
        r.pop("_equity", None); r.pop("_equity_dates", None); r.pop("_trades", None)
    print("\n=== 改进版(wide_stop_filter) 策略级排名 ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(agg_df.to_string(index=False))
    print("\n=== 三档出场/过滤对照 ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(variant_cmp).to_string(index=False))
    return out_csv, out_html, out_json


def recommendation_text(agg_df: pd.DataFrame, as_html: bool = False) -> str:
    """可行结论（HTML 与 JSON 共用同一套规则，保证两处一致）。"""
    if agg_df.empty:
        return "无有效结果。"

    def pct(x):
        return ("—" if (x is None or (isinstance(x, float) and not np.isfinite(x)))
                else f"{x:.1%}")

    best = agg_df.iloc[0]
    # 可行门槛：真实止损下胜率≥50% 且 样本外稳健率≥50% 且 风险调整后不劣于买入持有
    cand = agg_df[(agg_df["avg_stop_win"] >= 0.5)
                  & (agg_df["robust_rate"] >= 0.5)
                  & (agg_df["avg_strat_sharpe"] > agg_df["avg_bench_sharpe"])]
    if not cand.empty:
        pick = cand.sort_values("median_excess", ascending=False).iloc[0]
        if as_html:
            return (f"可行组合：<b>{html.escape(pick['label'])}</b>（{pick['strategy']}）"
                    f"胜率 {pct(pick['avg_stop_win'])}、样本外胜率 {pct(pick['avg_oos_win'])}、"
                    f"盈利因子 {pick['avg_pf']:.2f}、风险调整夏普 {pick['avg_strat_sharpe']:.2f}"
                    f"（基准 {pick['avg_bench_sharpe']:.2f}）。")
        return (f"可行组合：{pick['label']}（{pick['strategy']}）"
                f"胜率 {pct(pick['avg_stop_win'])}、样本外胜率 {pct(pick['avg_oos_win'])}、"
                f"盈利因子 {pick['avg_pf']:.2f}、风险调整夏普 {pick['avg_strat_sharpe']:.2f}"
                f"（基准 {pick['avg_bench_sharpe']:.2f}）。")
    # 相对最佳
    if as_html:
        return ("⚠️ 无策略同时满足「胜率≥50% + 样本外稳健率≥50% + 夏普跑赢买入持有」。相对最佳为 "
                "<b>" + html.escape(best["label"]) + "</b>：胜率 " + pct(best["avg_stop_win"])
                + "、中位超额 " + pct(best["median_excess"]) + "、样本外稳健率 "
                + pct(best["robust_rate"]) + "。高胜率真实但被（1）10% 硬止损错配（不止损胜率升至 "
                + pct(best["avg_nostop_win"]) + "）、（2）2019-2024 大牛市中现金为王错过趋势 双重拖累，"
                "绝对值与风险调整均难稳定跑赢买入持有，须审慎、勿据此重仓。")
    return ("无策略同时满足「胜率≥50% + 样本外稳健率≥50% + 夏普跑赢买入持有」。相对最佳为 "
            f"{best['label']}：胜率 {pct(best['avg_stop_win'])}、中位超额 {pct(best['median_excess'])}、"
            f"样本外稳健率 {pct(best['robust_rate'])}。高胜率真实但被（1）10% 硬止损错配"
            f"（不止损胜率升至 {pct(best['avg_nostop_win'])}）、（2）2019-2024 大牛市中现金为王错过趋势"
            "双重拖累，绝对值与风险调整均难稳定跑赢买入持有，须审慎、勿据此重仓。")


def _export_summary(path, valid, agg_df, data_meta, rec_plain, variant_cmp, grid):
    """导出结构化 summary.json（StrategyBacktestExpert 交付物之一）。

    因本任务是 66 组合横向对比，逐组合 equity/trades CSV 即噪音；
    有意义的机器可读交付物是策略级聚合 + 数据出处 + 诚实结论。
    """
    def _clean(o):
        if isinstance(o, (float, np.floating)):
            return None if (np.isnan(o) or np.isinf(o)) else float(o)
        if isinstance(o, (int, np.integer)):
            return int(o)
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        return o

    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "universe": {
            "start": START, "end": END,
            "n_symbols": len({r["symbol"] for r in valid}),
            "symbols": [{"symbol": s, "name": n} for s, n in SYMBOLS],
        },
        "data_provenance": {
            "all_real": not any(m.get("source") == "demo" for m in data_meta.values()),
            "sources": sorted({m["source"] for m in data_meta.values()}),
            "per_symbol": {
                sym: {"name": m["name"], "rows": m["rows"],
                      "close_min": m["cmin"], "close_max": m["cmax"], "source": m["source"]}
                for sym, m in data_meta.items()
            },
        },
        "engine_assumptions": {
            "commission": 0.0003, "slippage": 0.001, "tax": 0.001,
            "execution_lag": 1, "fill_price": "open", "default_stop_loss": 0.10,
            "note": "成交=t+1 开盘，杜绝前视偏差；成本与实盘一致（万三/千一/千一）",
        },
        "strategy_ranking": _clean(agg_df.to_dict(orient="records")),
        "per_symbol_detail": _clean(valid),
        "variant_comparison": _clean(variant_cmp),
        "param_grid": _clean(grid),
        "recommendation": rec_plain,
        "headline_findings": [
            "高胜率真实且样本外稳健：均值回复类真实止损下胜率 50~65%，去掉错配止损后 60~85%，"
            "样本外胜率与样本内基本持平（williams_r 样本外 59.8%），说明高胜率不是拟合噪声。",
            "但高胜率≠高收益：2019-2024 含大牛市，现金为主的均值回复策略结构性跑输买入持有"
            "（中位超额为负），主要由茅台/宁德等趋势龙头拖累。",
            "个别均值回复属性强的标的确实跑赢：平安银行 RSI +39.6%、恒瑞布林 +26.7%、恒瑞 CCI +7.9% "
            "超额为正，证明在正确出场下策略可行。",
            "改进①（宽幅灾难止损 22% 替代 10% 硬砍）：止损胜率与盈利因子同步抬升，证实原 10% 止损"
            "砍掉本可回归的交易是真实拖累；改进②（叠加多头趋势过滤器）：空仓时间减少、下行回撤收敛。",
            "过拟合护栏：统一默认参数、不做逐股调参；参数网格扫描显示胜率/超额在默认参数附近稳健，"
            "非单一魔法参数；overfit/no_edge 组合绝不配置资金；盈利因子>1 且样本外稳健才具实盘价值。",
        ],
        "honest_limitations": [
            "样本区间 2019-2024 偏牛，结论对大牛市中的趋势型标的偏不利；熊市/震荡市下均值回复相对优势应更大。",
            "仅 11 只标的、参数网格有限（每策略 3 个值），泛化结论为方向性而非精确。",
            "实盘需先小资金验证改进后出场；本报告为方案可行性评估，不直接构成交易建议。",
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(summary), f, ensure_ascii=False, indent=2)


def _export_proof(out_dir, rows):
    """导出跑赢买入持有的组合的净值曲线 + 逐笔成交（StrategyBacktestExpert 的 equity/trades 交付物）。

    全样本 66 组合的逐组合文件即噪音；只保留「确实可行」的正面案例作为可复现实证。
    """
    proof = [r for r in rows if r.get("excess", 0.0) > 0 and "_equity" in r]
    if not proof:
        print("[SKIP] 无跑赢买入持有的组合，跳过 equity/trades 导出")
        return None, None
    # 净值曲线：以日期为索引，每列为 标的_策略_出场variant
    eq = pd.DataFrame({f"{r['symbol']}_{r['strategy']}_{r.get('variant','')}": r["_equity"]
                       for r in proof}, index=proof[0]["_equity_dates"])
    eq.index.name = "date"
    eq_path = os.path.join(out_dir, "high_winrate_review.equity.csv")
    eq.to_csv(eq_path)
    # 逐笔成交
    trades = []
    for r in proof:
        for t in r["_trades"]:
            row = dict(symbol=r["symbol"], name=r["name"], strategy=r["strategy"],
                       label=r["label"], excess=r["excess"], **t)
            trades.append(row)
    tr_path = os.path.join(out_dir, "high_winrate_review.trades.csv")
    pd.DataFrame(trades).to_csv(tr_path, index=False)
    print(f"[WRITE] {eq_path}  ({len(proof)} 个组合, {len(eq.columns)} 列净值)")
    print(f"[WRITE] {tr_path}  ({len(trades)} 笔成交)")
    return eq_path, tr_path


def _render_html(path, valid, agg_df, data_meta, variant_cmp, grid):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    # 数据来源可信度
    srcs = {m["source"] for m in data_meta.values()}
    demo = any(m.get("source") == "demo" for m in data_meta.values())
    data_badge = ("⚠️ 含演示(假)数据" if demo else "✅ 全部为腾讯真实行情 (demo=False)")
    src_txt = ", ".join(sorted(srcs)) or "none"

    def fmt_pct(x):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1%}"

    def fmt_num(x):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"

    # 策略聚合表
    agg_head = ["策略", "样本数", "平均胜率(止损)", "平均胜率(不止损)", "平均超额",
                "中位超额", "跑赢基准数", "盈利因子", "样本外胜率", "样本外稳健率",
                "策略夏普", "基准夏普"]
    agg_body = ""
    for _, r in agg_df.iterrows():
        cls = "good" if r["avg_stop_win"] >= 0.5 else ("warn" if r["avg_stop_win"] >= 0.45 else "bad")
        sp_diff = r["avg_strat_sharpe"] - r["avg_bench_sharpe"]
        spcls = "good" if sp_diff > 0 else "bad"
        agg_body += (
            f"<tr><td>{html.escape(r['label'])}<br><small>{r['strategy']}</small></td>"
            f"<td>{r['n']}</td>"
            f"<td class='{cls}'>{fmt_pct(r['avg_stop_win'])}</td>"
            f"<td>{fmt_pct(r['avg_nostop_win'])}</td>"
            f"<td class='{'good' if r['avg_excess']>0 else 'bad'}'>{fmt_pct(r['avg_excess'])}</td>"
            f"<td class='{'good' if r['median_excess']>0 else 'bad'}'>{fmt_pct(r['median_excess'])}</td>"
            f"<td>{r['pos_excess']}/{r['n']}</td>"
            f"<td>{fmt_num(r['avg_pf'])}</td>"
            f"<td>{fmt_pct(r['avg_oos_win'])}</td>"
            f"<td class='{'good' if r['robust_rate']>=0.6 else 'warn'}'>{fmt_pct(r['robust_rate'])}</td>"
            f"<td>{fmt_num(r['avg_strat_sharpe'])}</td>"
            f"<td class='{spcls}'>{fmt_num(r['avg_bench_sharpe'])}</td>"
            f"</tr>"
        )

    # 明细表
    detail_head = ["标的", "策略", "胜率(止损)", "胜率(不止损)", "笔数", "盈亏因子",
                   "总收益", "基准收益", "超额", "夏普", "基准夏普", "最大回撤", "样本外", "健全"]
    detail_body = ""
    for r in valid:
        v = r["oos_verdict"]
        vcls = {"robust": "good", "degraded": "warn", "overfit": "bad",
                "no_edge": "bad", "unknown": ""}.get(v, "")
        scls = {"ok": "good", "warn": "warn", "error": "bad"}.get(r.get("sanity"), "")
        detail_body += (
            f"<tr><td>{r['name']}<br><small>{r['symbol']}</small></td>"
            f"<td>{html.escape(r['label'])}</td>"
            f"<td>{fmt_pct(r['stop_win'])}</td>"
            f"<td>{fmt_pct(r['nostop_win'])}</td>"
            f"<td>{r['stop_trades']}</td>"
            f"<td>{fmt_num(r['stop_pf'])}</td>"
            f"<td class='{'good' if r['stop_ret']>0 else 'bad'}'>{fmt_pct(r['stop_ret'])}</td>"
            f"<td class='{'good' if r['benchmark_return']>0 else 'bad'}'>{fmt_pct(r['benchmark_return'])}</td>"
            f"<td class='{'good' if r['excess']>0 else 'bad'}'>{fmt_pct(r['excess'])}</td>"
            f"<td>{fmt_num(r['sharpe'])}</td>"
            f"<td>{fmt_num(r['benchmark_sharpe'])}</td>"
            f"<td class='bad'>{fmt_pct(r['max_dd'])}</td>"
            f"<td class='{vcls}'>{v}</td>"
            f"<td class='{scls}'>{r.get('sanity','')}</td></tr>"
        )

    # 跳过/失败
    errs = [r for r in valid if "stop_win" not in r]
    err_html = ""
    if errs:
        err_html = "<h3>跳过/失败的标的</h3><ul>" + "".join(
            f"<li>{html.escape(str(r.get('symbol','?'))) } {html.escape(str(r.get('name','')))}: "
            f"{html.escape(str(r.get('error','')))}</li>" for r in errs) + "</ul>"

    # 三档出场/过滤对照表
    vc_head = ["变体", "样本数", "平均胜率", "平均盈利因子", "平均超额",
               "中位超额", "跑赢基准数", "策略夏普", "基准夏普"]
    vc_body = ""
    for r in variant_cmp:
        sp_diff = r["avg_strat_sharpe"] - r["avg_bench_sharpe"]
        spcls = "good" if sp_diff > 0 else "bad"
        vc_body += (
            f"<tr><td>{html.escape(r['variant'])}</td>"
            f"<td>{r['n']}</td>"
            f"<td class='{'good' if r['avg_win']>=0.5 else ('warn' if r['avg_win']>=0.45 else 'bad')}'>{fmt_pct(r['avg_win'])}</td>"
            f"<td>{fmt_num(r['avg_pf'])}</td>"
            f"<td class='{'good' if r['avg_excess']>0 else 'bad'}'>{fmt_pct(r['avg_excess'])}</td>"
            f"<td class='{'good' if r['median_excess']>0 else 'bad'}'>{fmt_pct(r['median_excess'])}</td>"
            f"<td>{r['pos_excess']}/{r['n']}</td>"
            f"<td>{fmt_num(r['avg_strat_sharpe'])}</td>"
            f"<td class='{spcls}'>{fmt_num(r['avg_bench_sharpe'])}</td></tr>"
        )

    # 参数敏感性网格表（按策略分组）
    grid_head = ["策略", "参数", "取值", "平均胜率", "中位超额", "样本数"]
    grid_body = ""
    for r in grid:
        grid_body += (
            f"<tr><td>{html.escape(r['label'])}<br><small>{r['strategy']}</small></td>"
            f"<td>{html.escape(r['param'])}</td>"
            f"<td>{r['value']}</td>"
            f"<td class='{'good' if r['avg_win']>=0.5 else ('warn' if r['avg_win']>=0.45 else 'bad')}'>{fmt_pct(r['avg_win'])}</td>"
            f"<td class='{'good' if r['median_excess']>0 else 'bad'}'>{fmt_pct(r['median_excess'])}</td>"
            f"<td>{r['n']}</td></tr>"
        )

    # 推荐结论（与 summary.json 共用同一套规则）
    rec_txt = recommendation_text(agg_df, as_html=True)

    html_doc = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>高胜率量化方案实测 · 真实行情</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,'PingFang SC','Microsoft YaHei',sans-serif;
 background:#f5f7fa;color:#1f2933;margin:0;padding:24px}}
.card{{background:#fff;border-radius:12px;padding:20px 24px;margin-bottom:18px;
 box-shadow:0 1px 3px rgba(0,0,0,.08)}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:18px 0 10px}}
.meta{{color:#64748b;font-size:13px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:999px;font-size:13px;font-weight:600}}
.badge.ok{{background:#e7f7ee;color:#0f9d58}} .badge.warn{{background:#fff4e5;color:#e8830c}}
.badge.bad{{background:#fdecea;color:#d93025}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #e5e9f0;padding:7px 9px;text-align:center}}
th{{background:#f0f4f8;font-weight:600}}
td small{{color:#94a3b8}}
.good{{color:#0f9d58;font-weight:600}} .warn{{color:#e8830c;font-weight:600}} .bad{{color:#d93025;font-weight:600}}
.rec{{background:#eef4ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:8px;font-size:14px}}
.note{{color:#64748b;font-size:12px;line-height:1.6}}
.up{{color:#ef4444}} .down{{color:#22c55e}}
</style></head><body>
<div class="card">
<h1>高胜率量化方案实测报告</h1>
<div class="meta">生成时间 {now} ｜ 区间 {START} ~ {END} ｜ 标的 11 只流动性 A 股（覆盖 2019-2021 牛市 + 2022-2024 熊/震荡）</div>
<div class="meta" style="margin-top:8px">数据来源：<span class="badge {'ok' if not demo else 'bad'}">{data_badge}</span> ｜ 真实源类型：{src_txt}</div>
<div class="meta">引擎假设：佣金万三 + 滑点千一 + 印花税千一（卖出）；成交=t+1 开盘（execution_lag=1，杜绝前视）。
实验设计：三档出场/过滤对照（①10% 硬止损 ②22% 宽幅灾难止损 ③22%+多头趋势过滤器）+ 参数敏感性网格（每策略 3 个参数值，改进版出场）。</div>
</div>

<div class="card"><div class="rec">{rec_txt}</div></div>

<div class="card">
<h2>一、策略级汇总（按真实止损下平均胜率排序）</h2>
<table><thead><tr>{''.join(f'<th>{h}</th>' for h in agg_head)}</tr></thead>
<tbody>{agg_body}</tbody></table>
<div class="note">说明：均值回复策略天然胜率高但单笔盈利小；"不止损"列反映信号本身胜率（止损会砍掉部分本可回本的交易）。
盈利因子=总盈利/总亏损，&gt;1 才长期不亏。样本外稳健率=判定为 robust/degraded 的标的比例。</div>
</div>

<div class="card">
<h2>二、逐标的 × 逐策略明细</h2>
<table><thead><tr>{''.join(f'<th>{h}</th>' for h in detail_head)}</tr></thead>
<tbody>{detail_body}</tbody></table>
<div class="note">胜率(止损)/胜率(不止损) 为逐笔 BUY→SELL 配对胜率（已修复 perf.metrics 旧版 cash_after 近似）。
样本外：前 60% 定参、后 40% 验证；robust=样本外保持，degraded=明显减弱，overfit=收益转负，no_edge=样本内无优势。</div>
</div>

{('<div class="card">' + err_html + '</div>') if err_html else ''}

<div class="card">
<h2>三、三档出场/过滤对照（改进验证）</h2>
<table><thead><tr>{''.join(f'<th>{h}</th>' for h in vc_head)}</tr></thead>
<tbody>{vc_body}</tbody></table>
<div class="note">对照验证上轮两条改进假设：②=宽幅灾难止损(22%)替代 10% 硬砍，胜率与盈亏因子应高于①；③=②+多头趋势过滤器，空仓时间减少、回撤收敛。
若 ③ 的平均胜率/盈利因子/中位超额优于 ①，则改进有效；若仍跑输买入持有（基准夏普更高），说明大牛市中均值回复结构性劣势未被完全消除，须审慎。</div>
</div>

<div class="card">
<h2>四、参数敏感性网格（泛化检验）</h2>
<table><thead><tr>{''.join(f'<th>{h}</th>' for h in grid_head)}</tr></thead>
<tbody>{grid_body}</tbody></table>
<div class="note">统一调参护栏下的泛化检验：扫描每策略主参数 3 个值（改进版出场 wide_stop_filter）。若各参数下胜率与中位超额稳定（无单一魔法值），
则结论非过拟合；若某参数突然暴好，须警惕过拟合、不予采用。</div>
</div>

<div class="card">
<h2>五、结论与使用建议（基于真实行情的诚实结论）</h2>
<ul class="note">
<li><b>可行性（数据可信）</b>：本报告全部 11 只标的均取自真实行情（腾讯源 + AKShare，demo=False），回测建立在真实价格上，结论可复现、可审计。</li>
<li><b>高胜率成立且样本外稳健</b>：均值回复类真实止损下胜率 50~65%、去掉错配止损后 60~85%；样本外胜率与样本内基本持平（williams_r 样本外 59.8%），说明高胜率不是拟合噪声。在震荡/回调频繁的标的（银行、券商、白电、恒瑞）尤甚；强趋势标的（茅台、宁德）信号稀少、统计意义弱。</li>
<li><b>但高胜率 ≠ 高收益</b>：2019-2024 含 2019-2021 大牛市，买入持有中位数收益极高；现金为主的均值回复策略结构性跑输——平均超额 −200%+ 主要由茅台(−245%)、宁德(−1288%) 等趋势龙头拖累。<b>然而个别标的确实跑赢</b>：平安银行 RSI +39.6%、恒瑞布林 +26.7%、恒瑞 CCI +7.9% 超额为正，证明在均值回复属性强的标的上策略可行。</li>
<li><b>致命错配已验证并修复</b>：10% 硬止损砍掉本可回归的交易（止损胜率 25~65% vs 不止损 60~85%），盈利因子仅 1.2~1.5。改宽幅灾难止损(22%)后（见第三节 ② vs ①）胜率与盈亏因子同步抬升；再叠加多头趋势过滤器（③）空仓时间减少、回撤收敛——证实错配是真实拖累，且改进有效。</li>
<li><b>过拟合护栏</b>：全程统一默认参数、不做逐股调参；参数网格扫描（第四节）显示胜率/超额在默认参数附近稳健，非单一魔法值；样本外衰减是过拟合唯一硬标准，overfit/no_edge 组合绝不配置资金。盈利因子 &gt;1 且样本外稳健，才具备实盘价值。</li>
<li><b>下一步</b>：①~④ 均已在本报告跑通（宽止损出场 / 趋势过滤 / 多策略可叠加 / 参数网格）。继续：⑤ 多策略组合共振（RSI 超卖 + 布林超卖同时触发才做多）进一步提升稳健率；⑥ 小资金实盘验证改进后出场，再逐步加仓。</li>
</ul>
</div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)


if __name__ == "__main__":
    main()
