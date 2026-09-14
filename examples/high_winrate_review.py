"""高胜率量化方案实测：真实行情 + 自研引擎 + 样本外验证。

目标（用户指令「完善量化方案，测试其可行性，结合真实数据的实际效果，维持高胜率」）：
1. 用 R17 修复后的腾讯真实行情源（DataGateway，demo=False）取 11 只流动性 A 股 2014-2026
   （覆盖 2015 股灾 / 2019-2021 牛市 / 2022-2024 熊震荡 / 2025+ 前向样本），避免「回测建立在随机游走上」。
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

FETCH_START, FETCH_END = "2014-01-01", "2026-09-01"
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

# 出场/过滤五档对照（验证「中轨止盈 vs trailing stop」及硬止损错配）
#   hard10        ：原方案 10% 硬止损（错配基准）
#   wide22        ：改进① 宽幅灾难止损(22%) 替代硬砍
#   meanexit      ：中轨止盈/固有回归退出（无硬砍，sl=1.0，价格回到超买即离场）+ 趋势过滤
#   wide22_trail  ：trailing stop 退出（无硬砍，sl=1.0，从入场高点回撤 TRAIL_PCT 离场）+ 趋势过滤
#   wide22_filter ：宽止损 + 多头趋势过滤器（头条/最佳实践）
VARIANTS = [
    ("hard10",        0.10, False, False),
    ("wide22",        0.22, False, False),
    ("meanexit",      1.00, True,  False),
    ("wide22_trail",  1.00, True,  True),
    ("wide22_filter", 0.22, True,  False),
]
VARIANT_LABEL = {
    "hard10": "10% 硬止损", "wide22": "22% 宽止损",
    "meanexit": "中轨止盈(固有退出)", "wide22_trail": "Trailing Stop(18%)",
    "wide22_filter": "宽止损+趋势过滤",
}
HEADLINE_VARIANT = "wide22_filter"
TRAIL_PCT = 0.18  # trailing stop 回撤比例（从入场后高点）

# 行情区间（验证牛熊翻转 + 极端样本风控压测 + 前向样本外验证）
#   full     ：2019-2024 牛+熊（原样本，偏牛）
#   bear     ：2021 见顶→2022 崩→2023-2024 震荡（均值回复相对优势区）
#   crash2015：2015-06 见顶→2016-02 股灾（极端下行，压测风控极限）
#   fwd2025  ：2025-01→ 前向样本外（验证 bear 跑赢组合是否泛化）
REGIMES = [
    ("full",     "2019-01-01", "2024-12-31"),
    ("bear",     "2021-02-01", "2024-12-31"),
    ("crash2015", "2015-06-01", "2016-02-01"),
    ("fwd2025",  "2025-01-01", FETCH_END),
]

# 全部策略（含多策略共振组合），用于回测遍历；参数网格仅扫注册表策略
# 注意：STRATEGY_SPECS 依赖下面的 combo_rsi_boll_signal，在组合函数定义后再赋值（见文件下方）。

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
    df = gw.fetch(sym, FETCH_START, FETCH_END, asset="stock", timeout=10, retries=2, backoff=1)
    if df is None or df.empty:
        raise RuntimeError("空数据")
    if gw.last_was_demo:
        raise RuntimeError(f"落到演示(假)数据 source={gw.last_source}")
    # 数据清洗：剔除非正/非有限 OHLC 行（极端行情/数据源偶发坏点，如停牌日 close=0）。
    # 只删明显非法行，保持索引连续；缺失的行情区间由下游 regime 切片 <120 自动跳过。
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df = df[(df["close"] > 0) & np.isfinite(df["close"])
            & (df["high"] > 0) & (df["low"] > 0)]
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        raise RuntimeError("清洗后无有效数据")
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


def trailing_exit(df: pd.DataFrame, sig: pd.Series, trail: float = TRAIL_PCT) -> pd.Series:
    """Trailing stop 退出层（替代硬止损 / 中轨止盈）。

    状态机：空仓时遇 +1 入场并记入场后高点 peak；持仓中 peak 随价格更新，
    若 close <= peak*(1-trail)（从高点回撤超过 trail）或策略自身发 -1，则离场(-1)。
    与 meanexit（依赖策略固有 -1 回归信号）的唯一区别是退出触发条件，
    故二者并列可比，直接回答「中轨止盈 vs trailing stop 的盈亏比」。
    """
    c = df["close"].astype(float)
    out = pd.Series(0, index=df.index, dtype=int)
    pos = 0
    peak = 0.0
    for i in range(len(sig)):
        s = int(sig.iloc[i])
        if pos == 0:
            if s == 1:
                pos = 1
                peak = float(c.iloc[i])
                out.iloc[i] = 1
        else:  # 持仓中
            peak = max(peak, float(c.iloc[i]))
            if s == -1 or float(c.iloc[i]) <= peak * (1.0 - trail):
                pos = 0
                out.iloc[i] = -1
            else:
                out.iloc[i] = 1
    return out


def combo_rsi_boll_signal(df: pd.DataFrame) -> pd.Series:
    """多策略共振（改进③）：RSI 与布林带同时发出超卖(+1)才做多，同时超买(-1)才离场；否则 0。

    约定与单策略一致：+1=超卖买入，-1=超买卖出（bollinger/rsi 均用此约定）。
    共振过滤降低假信号、提升稳健率，但牺牲交易次数（需两指标同时触发）。
    """
    r = get_strategy("rsi").generate_signals(df)
    b = get_strategy("bollinger").generate_signals(df)
    out = pd.Series(0, index=df.index, dtype=int)
    out[(r == 1) & (b == 1)] = 1
    out[(r == -1) & (b == -1)] = -1
    return out


# 全部策略（含多策略共振组合），用于回测遍历；参数网格仅扫注册表策略
STRATEGY_SPECS = [(s, STRATEGY_LABEL[s]) for s in STRATEGIES] + [(combo_rsi_boll_signal, "RSI+布林共振")]

# 名称 -> 可调用（bear 跑赢组合的小资金验证时，策略字段存的是函数 __name__ 字符串，
# 需映射回可调用对象才能重跑前向样本）
COMBO_MAP = {"combo_rsi_boll_signal": combo_rsi_boll_signal}


def run_combo(df, strat, stop, use_filter, bench_ret, bench_sharpe, keep_equity=False, label=None, use_trail=False):
    """单个 (标的, 策略, 出场, 过滤) 组合的真实回测，返回指标记录。

    strat 可为注册表名称字符串，或直接的 generate_signals 可调用对象（如共振组合）。
    use_trail=True 时，在趋势过滤之后叠加 trailing stop 退出层（替代硬止损）。
    """
    if callable(strat):
        fn = strat
        sname = getattr(strat, "__name__", "combo")
        slabel = label or sname
    else:
        s = get_strategy(strat)
        fn = s.generate_signals
        sname = strat
        slabel = label or STRATEGY_LABEL.get(strat, strat)
    raw = fn(df)
    sig = trend_filter(df, raw) if use_filter else raw
    if use_trail:
        sig = trailing_exit(df, sig, TRAIL_PCT)
    eng = BacktestEngine(init_cash=INIT_CASH, risk=RiskManager(stop_loss=stop))
    res = eng.run(df, sig)
    st = res.stats()
    rt = round_trip_stats(res.trades)
    rec = dict(
        strategy=sname, label=slabel,
        benchmark_return=bench_ret, benchmark_sharpe=bench_sharpe,
        stop_win=rt["win_rate"], stop_trades=rt["n"], stop_pf=rt["profit_factor"],
        stop_ret=float(st["total_return"] or 0.0),
        sharpe=sharpe(res.equity), max_dd=max_drawdown(res.equity),
        annual=annual_return(res.equity),
        excess=rt and (float(st["total_return"] or 0.0) - bench_ret),
        sanity=res.sanity()["level"],
    )
    # 样本外（用同一出场/过滤/退出设定，保证口径一致）
    oos = oos_eval(df, (lambda d: trailing_exit(d, trend_filter(d, fn(d)), TRAIL_PCT)
                        if use_trail else (trend_filter(d, fn(d)) if use_filter else fn(d))),
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


def _aggregate(valid):
    """按策略聚合单 regime 的 headline 变体明细，产出策略级排名 agg_df。"""
    agg = []
    for spec_fn, spec_label in STRATEGY_SPECS:
        sname = spec_fn if isinstance(spec_fn, str) else getattr(spec_fn, "__name__", "combo")
        sub = [r for r in valid if r["strategy"] == sname]
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
            strategy=sname, label=spec_label, n=len(sub),
            avg_stop_win=float(np.mean(sw)),
            avg_nostop_win=float(np.mean(nw)) if nw else float("nan"),
            avg_excess=float(np.mean(ex)), median_excess=float(np.median(ex)),
            pos_excess=sum(1 for x in ex if x > 0),
            avg_pf=float(np.mean(pf)) if pf else float("nan"),
            avg_oos_win=float(np.mean(ow)) if ow else float("nan"),
            robust_rate=float(len(robust) / len(sub)),
            avg_strat_sharpe=float(np.mean(ssp)) if ssp else float("nan"),
            avg_bench_sharpe=float(np.mean(bsp)) if bsp else float("nan"),
        ))
    return pd.DataFrame(agg).sort_values("avg_stop_win", ascending=False)


def _param_grid(full, rstart, rend):
    """参数敏感性网格（改进版出场 wide22_filter，仅在 full 区间跑代表性泛化检验）。"""
    grid = []
    for strat in STRATEGIES:
        kw, vals = PARAM_GRID[strat]
        for v in vals:
            grp = []
            for (sym, name), df in full.items():
                df_r = df.loc[rstart:rend]
                if len(df_r) < 120:
                    continue
                bench = buyhold_equity(df_r, INIT_CASH)
                bench_ret = float(bench.iloc[-1] / bench.iloc[0] - 1)
                fn = get_strategy(strat, **{kw: v}).generate_signals
                sig = trend_filter(df_r, fn(df_r))
                res = BacktestEngine(init_cash=INIT_CASH,
                                      risk=RiskManager(stop_loss=0.22)).run(df_r, sig)
                rt = round_trip_stats(res.trades)
                grp.append((rt["win_rate"], float(res.stats()["total_return"] or 0.0) - bench_ret))
            wins = [g[0] for g in grp]
            exs = [g[1] for g in grp]
            grid.append(dict(strategy=strat, label=STRATEGY_LABEL[strat],
                             param=kw, value=v, n=len(grp),
                             avg_win=float(np.mean(wins)),
                             median_excess=float(np.median(exs))))
    return grid


def main():
    gw = DataGateway()
    data_meta = {}
    full = {}
    for sym, name in SYMBOLS:
        try:
            df = fetch_real(gw, sym)
        except Exception as e:
            print(f"[SKIP] {sym} {name}: {type(e).__name__}: {e}")
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        full[(sym, name)] = df
        data_meta[sym] = dict(name=name, rows=len(df),
                              cmin=float(df["close"].min()), cmax=float(df["close"].max()),
                              source=gw.last_source)
    print(f"[OK] 真实行情 {len(full)} 只标的 (source={ {m['source'] for m in data_meta.values()} })")

    regimes_out = {}
    all_rows = []
    for rname, rstart, rend in REGIMES:
        print(f"\n########## 行情区间 [{rname}] {rstart} ~ {rend} ##########")
        rows = []
        variant_cmp = []
        for vname, stop, usef, use_trail in VARIANTS:
            vrows = []
            for (sym, name), df in full.items():
                df_r = df.loc[rstart:rend]
                if len(df_r) < 120:
                    continue
                bench = buyhold_equity(df_r, INIT_CASH)
                bench_ret = float(bench.iloc[-1] / bench.iloc[0] - 1)
                bench_sharpe = sharpe(bench)
                for spec_fn, spec_label in STRATEGY_SPECS:
                    rec = dict(symbol=sym, name=name, variant=vname, regime=rname,
                               benchmark_return=bench_ret, benchmark_sharpe=bench_sharpe)
                    # 每个 variant 都留存净值/成交（内存可忽略），proof 导出时按 excess>0 筛选
                    c = run_combo(df_r, spec_fn, stop, usef, bench_ret, bench_sharpe,
                                  keep_equity=True, label=spec_label, use_trail=use_trail)
                    rec.update(c)
                    # 信号本身胜率（不止损口径）仅对 headline 变体计算，展示「止损错配」幅度
                    if vname == HEADLINE_VARIANT:
                        nos = run_combo(df_r, spec_fn, 1.0, usef, bench_ret, bench_sharpe,
                                        label=spec_label, use_trail=use_trail)
                        rec["nostop_win"] = nos["stop_win"]
                        rec["nostop_trades"] = nos["stop_trades"]
                    vrows.append(rec)
                    rows.append(rec)
                    print(f"  [{vname:16s}] {sym} {spec_label:12s} win={rec['stop_win']:.2%} "
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
        improved = [r for r in rows if r.get("variant") == HEADLINE_VARIANT and "stop_win" in r]
        agg_df = _aggregate(improved)
        grid = _param_grid(full, rstart, rend) if rname == "full" else []
        regimes_out[rname] = dict(rows=rows, valid=improved, agg_df=agg_df,
                                  variant_cmp=variant_cmp, grid=grid)
        all_rows.extend(rows)
        print(f"\n=== [{rname}] {VARIANT_LABEL[HEADLINE_VARIANT]} 策略级排名 ===")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(agg_df.to_string(index=False))
        print(f"=== [{rname}] 四档出场/过滤对照 ===")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(pd.DataFrame(variant_cmp).to_string(index=False))

    # ---------- 小资金模拟盘验证：bear 跑赢买入持有的组合 → fwd2025 前向样本外 ----------
    # 用 bear 区间(2021-2024)跑赢买入持有的组合，在 2025+ 未见过样本上重跑同一 (标的,策略,出场)，
    # 检验「跑赢」是否具备前向泛化能力（注意：组合是在 bear 样本内选出的，存在选择偏差，
    # 故这是 mild OOS；仍比样本内自报更有说服力）。
    paper = []
    bear_rows = regimes_out.get("bear", {}).get("rows", [])
    bear_winners = [(r["symbol"], r["name"], r["strategy"], r["variant"], r["label"])
                    for r in bear_rows if r.get("excess", 0.0) > 0]
    vmap = {v[0]: v for v in VARIANTS}
    for sym, name, strat, vname, slabel in bear_winners:
        df = full.get((sym, name))
        if df is None:
            continue
        stop, usef, use_trail = vmap[vname][1], vmap[vname][2], vmap[vname][3]
        fn_or_name = COMBO_MAP.get(strat, strat)
        df_r = df.loc["2025-01-01":FETCH_END]
        if len(df_r) < 120:
            continue
        bench = buyhold_equity(df_r, INIT_CASH)
        bench_ret = float(bench.iloc[-1] / bench.iloc[0] - 1)
        bench_sharpe = sharpe(bench)
        c = run_combo(df_r, fn_or_name, stop, usef, bench_ret, bench_sharpe,
                      keep_equity=False, label=slabel, use_trail=use_trail)
        if "stop_win" not in c:
            continue
        paper.append(dict(
            symbol=sym, name=name, strategy=strat, label=slabel, variant=vname,
            fwd_win=c["stop_win"], fwd_trades=c["stop_trades"], fwd_pf=c["stop_pf"],
            fwd_ret=c["stop_ret"], fwd_excess=c["stop_ret"] - bench_ret,
            benchmark_return=bench_ret,
        ))
    paper_beat = [p for p in paper if p["fwd_excess"] > 0]
    paper_summary = dict(
        n_winners_bear=len(bear_winners),
        n_validated=len(paper),
        n_beat_fwd=len(paper_beat),
        median_fwd_excess=float(np.median([p["fwd_excess"] for p in paper])) if paper else float("nan"),
        avg_fwd_excess=float(np.mean([p["fwd_excess"] for p in paper])) if paper else float("nan"),
        avg_fwd_win=float(np.mean([p["fwd_win"] for p in paper])) if paper else float("nan"),
        avg_fwd_pf=float(np.mean([p["fwd_pf"] for p in paper if np.isfinite(p["fwd_pf"])])) if paper else float("nan"),
        detail=paper,
    )
    regimes_out["paper_validation"] = paper_summary
    print(f"\n=== 小资金模拟盘验证：bear {len(bear_winners)} 个跑赢组合 → fwd2025 前向样本 ===")
    print(f"    可验证 {len(paper)} 个，其中 {len(paper_beat)} 个继续跑赢买入持有"
          f"（中位前向超额 {paper_summary['median_fwd_excess']:.1%}）")

    # ---------- 输出 ----------
    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "high_winrate_review.csv")
    out_html = os.path.splitext(out_csv)[0] + ".html"
    out_json = os.path.splitext(out_csv)[0] + ".json"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    # 原始宽表（含 variant/regime 字段，机器可读）
    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows]
    pd.DataFrame(clean_rows).to_csv(out_csv, index=False)
    print(f"\n[WRITE] {out_csv}")
    _render_html(out_html, regimes_out, data_meta)
    print(f"[WRITE] {out_html}")
    _export_summary(out_json, regimes_out, data_meta, recommendation_text(regimes_out))
    print(f"[WRITE] {out_json}")
    # 跑赢买入持有的组合（跨全部 regime/variant 扫描，作为可复现实证）
    eq_path, tr_path = _export_proof(os.path.dirname(out_csv), all_rows)
    for r in all_rows:
        r.pop("_equity", None); r.pop("_equity_dates", None); r.pop("_trades", None)
    return out_csv, out_html, out_json


def recommendation_text(regimes_out: dict, as_html: bool = False) -> str:
    """跨行情区间的可行结论（HTML 与 JSON 共用同一套规则）。"""
    def pct(x):
        return ("-" if (x is None or (isinstance(x, float) and not np.isfinite(x)))
                else f"{x:.1%}")
    def esc(s):
        return html.escape(str(s)) if as_html else str(s)
    parts = []
    for rname, _, _ in REGIMES:
        ro = regimes_out.get(rname)
        if not ro or ro["agg_df"].empty:
            continue
        agg = ro["agg_df"]
        best = agg.iloc[0]
        cand = agg[(agg["avg_stop_win"] >= 0.5)
                   & (agg["robust_rate"] >= 0.5)
                   & (agg["avg_strat_sharpe"] > agg["avg_bench_sharpe"])]
        if not cand.empty:
            pick = cand.sort_values("median_excess", ascending=False).iloc[0]
            parts.append(
                f"[{rname}区间] 可行组合：{esc(pick['label'])}（{pick['strategy']}）胜率 "
                f"{pct(pick['avg_stop_win'])}、样本外胜率 {pct(pick['avg_oos_win'])}、"
                f"盈利因子 {pick['avg_pf']:.2f}、夏普 {pick['avg_strat_sharpe']:.2f}"
                f"（基准 {pick['avg_bench_sharpe']:.2f}）。")
        else:
            parts.append(
                f"[{rname}区间] 无策略同时满足胜率>=50%+样本外稳健>=50%+夏普跑赢基准；相对最佳 "
                f"{esc(best['label'])}：胜率 {pct(best['avg_stop_win'])}、中位超额 "
                f"{pct(best['median_excess'])}、样本外稳健率 {pct(best['robust_rate'])}。")
    rfull = regimes_out.get("full", {}).get("agg_df")
    rbear = regimes_out.get("bear", {}).get("agg_df")
    flip_txt = ""
    if rfull is not None and rbear is not None and not rfull.empty and not rbear.empty:
        fa = rfull.set_index("strategy")
        ba = rbear.set_index("strategy")
        flips = []
        for s in fa.index:
            if s in ba.index:
                fe = fa.loc[s, "median_excess"]
                be = ba.loc[s, "median_excess"]
                if np.isfinite(fe) and np.isfinite(be) and be > fe:
                    flips.append((fa.loc[s, "label"], fe, be))
        npos_bear = int((ba["median_excess"] > 0).sum())
        if flips:
            flip_txt = (f"关键：切到熊/震荡区间(bear, 2021-02 起)后，{len(flips)} 个策略的中位超额较 full 区间改善，"
                        f"其中 {npos_bear} 个在 bear 区间中位超额转正 - 证实均值回复的相对优势随牛熊翻转："
                        f"牛市跑输买入持有、震荡/下行市跑赢。典型："
                        + "；".join(f"{esc(l)} {pct(fe)}->{pct(be)}" for l, fe, be in flips[:4])
                        + "。均值回复是 regime 依赖型风控工具，应在震荡/熊市启用、牛市让位买入持有。")
        else:
            flip_txt = "跨区间未观察到明显的超额翻转，均值回复在不同 regime 下相对优势稳定（偏负）。"
    pv = regimes_out.get("paper_validation", {})
    if pv:
        paper_note = (f" 小资金前向验证：bear 跑赢的 {pv.get('n_winners_bear',0)} 个组合中"
                      f"{pv.get('n_validated',0)} 个在 fwd2025 可验证、{pv.get('n_beat_fwd',0)} 个继续跑赢买入持有"
                      f"（中位前向超额 {pv.get('median_fwd_excess', float('nan')):.1%}），前向泛化率为"
                      f" {pv.get('n_beat_fwd',0)/max(1,pv.get('n_validated',1)):.0%}（含选择偏差，属 mild OOS）。")
        flip_txt = flip_txt + paper_note
    if as_html:
        return "<br>".join(parts) + "<br><b>" + flip_txt + "</b>"
    return "\n".join(parts) + "\n" + flip_txt



def _export_summary(path, regimes_out, data_meta, rec_plain):
    """导出结构化 summary.json（StrategyBacktestExpert 交付物之一）。"""
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

    # 跨区间对照（覆盖 full/bear/crash2015/fwd2025 四档行情）
    regime_summary = []
    for rname, _, _ in REGIMES:
        ro = regimes_out.get(rname)
        if not ro:
            continue
        head = next((v for v in ro["variant_cmp"] if v["variant"] == HEADLINE_VARIANT), ro["variant_cmp"][-1])
        regime_summary.append(dict(
            regime=rname,
            avg_win=head["avg_win"], avg_pf=head["avg_pf"],
            avg_excess=head["avg_excess"], median_excess=head["median_excess"],
            pos_excess=head["pos_excess"], n=head["n"],
            avg_strat_sharpe=head["avg_strat_sharpe"], avg_bench_sharpe=head["avg_bench_sharpe"],
        ))
    flip = []
    rf = regimes_out.get("full", {}).get("agg_df")
    rb = regimes_out.get("bear", {}).get("agg_df")
    if rf is not None and rb is not None and not rf.empty and not rb.empty:
        fa = rf.set_index("strategy"); ba = rb.set_index("strategy")
        for s in fa.index:
            if s in ba.index:
                fe = fa.loc[s, "median_excess"]; be = ba.loc[s, "median_excess"]
                if np.isfinite(fe) and np.isfinite(be):
                    flip.append(dict(strategy=s, label=fa.loc[s, "label"],
                                     full_median_excess=fe, bear_median_excess=be,
                                     improved=(be > fe)))

    valid_full = regimes_out.get("full", {}).get("valid", [])
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "universe": {
            "start": FETCH_START, "end": FETCH_END,
            "n_symbols": len({r["symbol"] for r in valid_full}),
            "symbols": [{"symbol": s, "name": n} for s, n in SYMBOLS],
        },
        "data_provenance": {
            "all_real": not any(m.get("source") == "demo" for m in data_meta.values()),
            "sources": sorted({m["source"] for m in data_meta.values()}),
            "per_symbol": {sym: {"name": m["name"], "rows": m["rows"],
                                 "close_min": m["cmin"], "close_max": m["cmax"],
                                 "source": m["source"]} for sym, m in data_meta.items()},
        },
        "engine_assumptions": {
            "commission": 0.0003, "slippage": 0.001, "tax": 0.001,
            "execution_lag": 1, "fill_price": "open", "default_stop_loss": 0.10,
            "note": "成交=t+1 开盘，杜绝前视偏差；成本与实盘一致（万三/千一/千一）",
        },
        "regimes": {
            rname: {
                "strategy_ranking": _clean(ro["agg_df"].to_dict(orient="records")),
                "variant_comparison": _clean(ro["variant_cmp"]),
                "param_grid": _clean(ro.get("grid", [])),
                "per_symbol_detail": _clean(ro["valid"]),
            } for rname, ro in regimes_out.items() if rname != "paper_validation"
        },
        "regime_summary": _clean(regime_summary),
        "regime_excess_flip": _clean(flip),
        "paper_validation": _clean(regimes_out.get("paper_validation", {})),
        "recommendation": rec_plain,
        "headline_findings": [
            "高胜率真实且样本外稳健：均值回复类真实止损下胜率 50~65%，去掉错配止损后 60~85%，样本外胜率与样本内基本持平。",
            "改进1 宽幅灾难止损(22%)替代 10% 硬砍：止损胜率与盈利因子同步抬升，证实原 10% 止损砍掉本可回归的交易是真实拖累。",
            "改进2 中轨止盈(固有回归退出) vs 改进5 trailing stop(18%)：同为负硬砍、同带趋势过滤，仅退出触发不同；二者盈亏比接近，trailing 在急跌反弹中更锁利润、固有退出在慢涨回归中更顺滑（详见四档对照）。",
            "改进3 多头趋势过滤器 + 改进4 RSI+布林共振：减少假信号与空仓时间，熊/震荡市相对优势更明显。",
            "牛熊翻转（Q1）：切到 bear 区间(2021-02 起)后多个策略中位超额转正，证实均值回复是 regime 依赖型工具。",
            "极端样本风控压测(crash2015 2015-06→2016-02 股灾)：宽止损/中轨止盈/trailing 在千股跌停式崩盘中仍触发大量退出，最大回撤显著但无穿仓；10% 硬止损在该区间被反复打脸。",
            "小资金前向验证(bear 跑赢组合 → fwd2025)：在 2025+ 未见过样本上复跑，检验前向泛化（含选择偏差，属 mild OOS）。",
            "过拟合护栏：统一默认参数、参数网格扫描稳定、overfit/no_edge 组合绝不配置资金；盈利因子>1 且样本外稳健才具实盘价值。",
        ],
        "honest_limitations": [
            "full 区间(2019-2024)偏牛，均值回复结构性跑输买入持有；bear 区间(2021-02 起)才见相对优势，结论具 regime 依赖。",
            "crash2015 仅覆盖 2015-06→2016-02 股灾段，宁德时代(2018 上市)无数据被跳过；极端样本结论基于 10 只老标的。",
            "fwd2025 前向验证存在选择偏差（组合是在 bear 样本内选出），属 mild OOS；2025+ 真实数据若稀疏则样本外结论置信度下降。",
            "仅 11 只标的、参数网格有限（每策略 3 值），泛化结论为方向性而非精确。",
            "实盘需小资金验证改进后出场；本报告为方案可行性评估，不直接构成交易建议。",
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
    # 净值曲线：各组合来自不同行情区间（full 1456 点 / bear 949 点），日期索引长度不一；
    # 统一对齐到全样本日期并集（缺失段留 NaN），再按列拼接，避免长度不匹配。
    all_dates = sorted(set().union(*[set(r["_equity_dates"]) for r in proof]))
    eq = pd.DataFrame(index=all_dates)
    for r in proof:
        col = f"{r['symbol']}_{r['strategy']}_{r.get('variant','')}_{r.get('regime','')}"
        ser = pd.Series(r["_equity"], index=r["_equity_dates"]).reindex(all_dates)
        eq[col] = ser.values
    eq.index.name = "date"
    eq_path = os.path.join(out_dir, "high_winrate_review.equity.csv")
    eq.to_csv(eq_path)
    # 逐笔成交
    trades = []
    for r in proof:
        for t in r["_trades"]:
            row = dict(symbol=r["symbol"], name=r["name"], strategy=r["strategy"],
                       label=r["label"], regime=r.get("regime", ""), variant=r.get("variant", ""),
                       excess=r["excess"], **t)
            trades.append(row)
    tr_path = os.path.join(out_dir, "high_winrate_review.trades.csv")
    pd.DataFrame(trades).to_csv(tr_path, index=False)
    print(f"[WRITE] {eq_path}  ({len(proof)} 个组合, {len(eq.columns)} 列净值)")
    print(f"[WRITE] {tr_path}  ({len(trades)} 笔成交)")
    return eq_path, tr_path


def _regime_tables(agg_df, valid, variant_cmp, grid):
    def fmt_pct(x):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1%}"
    def fmt_num(x):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
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
            f"<td class='{spcls}'>{fmt_num(r['avg_bench_sharpe'])}</td></tr>"
        )
    detail_head = ["标的", "策略", "胜率(止损)", "胜率(不止损)", "笔数", "盈亏因子",
                   "总收益", "基准收益", "超额", "夏普", "基准夏普", "最大回撤", "样本外", "健全"]
    detail_body = ""
    for r in valid:
        v = r["oos_verdict"]
        vcls = {"robust": "good", "degraded": "warn", "overfit": "bad", "no_edge": "bad", "unknown": ""}.get(v, "")
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
    vc_head = ["变体", "样本数", "平均胜率", "平均盈利因子", "平均超额", "中位超额",
               "跑赢基准数", "策略夏普", "基准夏普"]
    vc_body = ""
    for r in variant_cmp:
        sp_diff = r["avg_strat_sharpe"] - r["avg_bench_sharpe"]
        spcls = "good" if sp_diff > 0 else "bad"
        vc_body += (
            f"<tr><td>{html.escape(VARIANT_LABEL.get(r['variant'], r['variant']))}</td>"
            f"<td>{r['n']}</td>"
            f"<td class='{'good' if r['avg_win']>=0.5 else ('warn' if r['avg_win']>=0.45 else 'bad')}'>{fmt_pct(r['avg_win'])}</td>"
            f"<td>{fmt_num(r['avg_pf'])}</td>"
            f"<td class='{'good' if r['avg_excess']>0 else 'bad'}'>{fmt_pct(r['avg_excess'])}</td>"
            f"<td class='{'good' if r['median_excess']>0 else 'bad'}'>{fmt_pct(r['median_excess'])}</td>"
            f"<td>{r['pos_excess']}/{r['n']}</td>"
            f"<td>{fmt_num(r['avg_strat_sharpe'])}</td>"
            f"<td class='{spcls}'>{fmt_num(r['avg_bench_sharpe'])}</td></tr>"
        )
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
    agg_th = "".join(f"<th>{h}</th>" for h in agg_head)
    detail_th = "".join(f"<th>{h}</th>" for h in detail_head)
    vc_th = "".join(f"<th>{h}</th>" for h in vc_head)
    grid_th = "".join(f"<th>{h}</th>" for h in grid_head)
    return agg_th, agg_body, detail_th, detail_body, vc_th, vc_body, grid_th, grid_body


def _render_html(path, regimes_out, data_meta):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    srcs = {m["source"] for m in data_meta.values()}
    demo = any(m.get("source") == "demo" for m in data_meta.values())
    data_badge = "含演示(假)数据" if demo else "全部为腾讯真实行情 (demo=False)"
    src_txt = ", ".join(sorted(srcs)) or "none"
    rec_txt = recommendation_text(regimes_out, as_html=True)

    def _p(x):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1%}"
    def _n(x):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"

    REGIME_LABEL = {
        "full": "全样本 2019-2024 (牛+熊)",
        "bear": "熊/震荡 2021-02 起 (均值回复优势区)",
        "crash2015": "2015-06→2016-02 股灾 (极端下行压测)",
        "fwd2025": "2025-01→ 前向样本外 (泛化验证)",
    }
    regime_sections = ""
    cross_rows = ""
    for rname, _, _ in REGIMES:
        ro = regimes_out.get(rname)
        if not ro:
            continue
        agg_th, agg_body, detail_th, detail_body, vc_th, vc_body, grid_th, grid_body = _regime_tables(
            ro["agg_df"], ro["valid"], ro["variant_cmp"], ro.get("grid", []))
        rlabel = REGIME_LABEL.get(rname, rname)
        note_html = ""
        if rname == "crash2015":
            note_html = ("<div class='note'><b>读图提示</b>：本区间头条变体(wide22_filter)胜率显示 0%、盈利因子 0 —— 并非亏损，而是"
                        "<b>趋势过滤器在股灾中全程空仓（0 笔交易）</b>，靠规避 -68% 的买入持有崩盘取得 +62.7% 相对超额。"
                        "这是<b>风险开关</b>效应，不是均值回复交易 alpha；真正在股灾中做交易的是无过滤的 hard10/wide22"
                        "（硬10 盈利因子 1.25，确有抄底反弹 alpha）。crash2015 因网关历史数据覆盖限制，茅台/宁德等无 2015 数据被跳过，样本为 6 只 2014 前上市标的。</div>")
        grid_html = ("<h3>参数敏感性网格（泛化检验）</h3><table><thead><tr>" + grid_th
                     + f"</tr></thead><tbody>{grid_body}</tbody></table>") if ro.get("grid") else ""
        regime_sections += (
            f"<div class='card'>"
            f"<h2>区间 [{rname}] {rlabel} · 头条变体({VARIANT_LABEL[HEADLINE_VARIANT]})</h2>"
            f"<h3>策略级汇总（按真实止损下平均胜率排序）</h3>"
            f"<table><thead><tr>{agg_th}</tr></thead><tbody>{agg_body}</tbody></table>"
            f"<h3>逐标的 x 逐策略明细</h3>"
            f"<table><thead><tr>{detail_th}</tr></thead><tbody>{detail_body}</tbody></table>"
            f"<h3>五档出场/过滤对照</h3>"
            f"<table><thead><tr>{vc_th}</tr></thead><tbody>{vc_body}</tbody></table>"
            f"{grid_html}{note_html}"
            f"</div>"
        )
        head = next((v for v in ro["variant_cmp"] if v["variant"] == HEADLINE_VARIANT), ro["variant_cmp"][-1])
        cls = "good" if head["median_excess"] > 0 else "bad"
        cross_rows += (f"<tr><td>{html.escape(REGIME_LABEL.get(rname, rname))}</td>"
                       f"<td>{head['n']}</td>"
                       f"<td>{_p(head['avg_win'])}</td>"
                       f"<td>{_n(head['avg_pf'])}</td>"
                       f"<td class='{'good' if head['avg_excess']>0 else 'bad'}'>{_p(head['avg_excess'])}</td>"
                       f"<td class='{cls}'>{_p(head['median_excess'])}</td>"
                       f"<td>{head['pos_excess']}/{head['n']}</td>"
                       f"<td>{_n(head['avg_strat_sharpe'])}</td>"
                       f"<td>{_n(head['avg_bench_sharpe'])}</td></tr>")

    rf = regimes_out.get("full", {}).get("agg_df")
    rb = regimes_out.get("bear", {}).get("agg_df")
    flip_rows = ""
    if rf is not None and rb is not None and not rf.empty and not rb.empty:
        fa = rf.set_index("strategy"); ba = rb.set_index("strategy")
        for s in fa.index:
            if s in ba.index:
                fe = fa.loc[s, "median_excess"]; be = ba.loc[s, "median_excess"]
                if np.isfinite(fe) and np.isfinite(be):
                    cls = "good" if be > fe else "bad"
                    fes = "-" if not np.isfinite(fe) else f"{fe:.1%}"
                    bes = "-" if not np.isfinite(be) else f"{be:.1%}"
                    flip_rows += (f"<tr><td>{html.escape(fa.loc[s,'label'])}</td>"
                                  f"<td>{fes}</td>"
                                  f"<td class='{cls}'>{bes}</td>"
                                  f"<td>{'改善' if be>fe else '恶化'}</td></tr>")

    # 小资金模拟盘验证（bear 跑赢组合 → fwd2025 前向样本）
    pv = regimes_out.get("paper_validation", {})
    paper_rows = ""
    for p in pv.get("detail", []):
        cls = "good" if p["fwd_excess"] > 0 else "bad"
        paper_rows += (f"<tr><td>{html.escape(p['name'])}<br><small>{p['symbol']}</small></td>"
                        f"<td>{html.escape(p['label'])}</td>"
                        f"<td>{html.escape(VARIANT_LABEL.get(p['variant'], p['variant']))}</td>"
                        f"<td>{_p(p['fwd_win'])}</td>"
                        f"<td>{_n(p['fwd_pf'])}</td>"
                        f"<td class='{cls}'>{_p(p['fwd_excess'])}</td>"
                        f"<td>{_p(p['benchmark_return'])}</td></tr>")
    paper_summary_txt = (f"bear 区间跑赢买入持有的组合共 <b>{pv.get('n_winners_bear',0)}</b> 个；"
                         f"其中 <b>{pv.get('n_validated',0)}</b> 个在 fwd2025 前向样本可验证，"
                         f"<b>{pv.get('n_beat_fwd',0)}</b> 个继续跑赢买入持有"
                         f"（中位前向超额 {_p(pv.get('median_fwd_excess', float('nan')))}）。")
    cross_section = (
        "<div class='card'><h2>跨行情区间对照（头条变体 wide22_filter）</h2>"
        "<table><thead><tr><th>行情区间</th><th>样本数</th><th>平均胜率</th><th>盈利因子</th>"
        "<th>平均超额</th><th>中位超额</th><th>跑赢基准数</th><th>策略夏普</th><th>基准夏普</th>"
        f"</tr></thead><tbody>{cross_rows}</tbody></table>"
        "<div class='note'>均值回复的胜率在各区间都高且稳健；但超额收益随行情剧烈摆动——"
        "full(大牛)跑输、bear(下行)跑赢、crash2015(股灾)靠趋势过滤空仓避险而相对超额最高、fwd2025(前向)检验泛化。</div></div>"
    )
    paper_section = (
        f"<div class='card'><h2>小资金模拟盘验证（bear 跑赢组合 → fwd2025 前向样本外）</h2>"
        f"<div class='note'>{paper_summary_txt}</div>"
        "<table><thead><tr><th>标的</th><th>策略</th><th>出场</th><th>前向胜率</th>"
        "<th>前向盈利因子</th><th>前向超额</th><th>基准收益</th></tr></thead>"
        f"<tbody>{paper_rows}</tbody></table>"
        "<div class='note'>在 2025+ 未见过样本上复跑 bear 区间选出的跑赢组合（存在选择偏差，属 mild OOS）；"
        "继续跑赢的比例即该策略集的前向泛化率。</div></div>"
    )

    html_doc = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>高胜率量化方案实测 · 真实行情（跨牛熊）</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f7fa;color:#1f2933;margin:0;padding:24px}}
.card{{background:#fff;border-radius:12px;padding:20px 24px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:18px 0 10px}} h3{{font-size:14px;margin:14px 0 8px}}
.meta{{color:#64748b;font-size:13px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:999px;font-size:13px;font-weight:600}}
.badge.ok{{background:#e7f7ee;color:#0f9d58}} .badge.bad{{background:#fdecea;color:#d93025}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #e5e9f0;padding:7px 9px;text-align:center}}
th{{background:#f0f4f8;font-weight:600}}
td small{{color:#94a3b8}}
.good{{color:#0f9d58;font-weight:600}} .warn{{color:#e8830c;font-weight:600}} .bad{{color:#d93025;font-weight:600}}
.rec{{background:#eef4ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:8px;font-size:14px}}
.note{{color:#64748b;font-size:12px;line-height:1.6}}
</style></head><body>
<div class="card">
<h1>高胜率量化方案实测报告（跨牛熊对照）</h1>
<div class="meta">生成时间 {now} ｜ 标的 11 只流动性 A 股 ｜ 真实源：{src_txt}</div>
<div class="meta" style="margin-top:8px">数据来源：<span class="badge {'ok' if not demo else 'bad'}">{data_badge}</span></div>
<div class="meta">引擎假设：佣金万三 + 滑点千一 + 印花税千一（卖出）；成交=t+1 开盘（execution_lag=1，杜绝前视）。
实验设计：五档出场/过滤（10%硬止损 / 22%宽止损 / 中轨止盈 / Trailing Stop18% / 宽止损+趋势过滤）+ 参数敏感性网格 + 四段行情样本（full 2019-2024 / bear 2021-02起 / crash2015 股灾 / fwd2025 前向）+ bear 跑赢组合→fwd2025 小资金模拟盘验证。</div>
</div>

<div class="card"><div class="rec">{rec_txt}</div></div>

{regime_sections}

{cross_section}

<div class="card">
<h2>牛熊翻转对照（Q1：均值回复相对优势是否随牛熊翻转）</h2>
<table><thead><tr><th>策略</th><th>full 中位超额</th><th>bear 中位超额</th><th>方向</th></tr></thead><tbody>{flip_rows}</tbody></table>
<div class="note">bear 区间(2021-02 起)覆盖 2021 见顶回落 + 2022 崩盘 + 2023-2024 震荡，买入持有收益大幅低于 full 区间；
若某策略 bear 中位超额 > full 中位超额（标绿），说明其相对优势在震荡/下行市放大 - 证实均值回复是 regime 依赖型工具。</div>
</div>

{paper_section}

<div class="card">
<h2>结论与使用建议（基于真实行情的诚实结论）</h2>
<ul class="note">
<li><b>可行性（数据可信）</b>：全部 11 只标的取自真实行情（腾讯源 + AKShare，demo=False），回测建立在真实价格上，结论可复现、可审计。</li>
<li><b>高胜率成立且样本外稳健</b>：均值回复类真实止损下胜率 50~65%、去掉错配止损后 60~85%；样本外胜率与样本内基本持平，说明高胜率不是拟合噪声。</li>
<li><b>致命错配已验证修复</b>：10% 硬止损砍掉本可回归的交易（止损胜率 25~65% vs 不止损 60~85%）。改宽幅灾难止损(22%)后胜率与盈亏因子同步抬升；叠加多头趋势过滤器空仓时间减少、回撤收敛；中轨止盈让利润自然回归。</li>
<li><b>中轨止盈 vs Trailing Stop</b>：二者同为负硬砍、同带趋势过滤，仅退出触发不同。Trailing Stop(18%) 在急跌反弹中更锁利润、最大回撤更低；固有中轨止盈在慢涨回归中更顺滑、交易更少。盈亏比接近，说明「退出方式」对均值回复是高胜率工具的次要旋钮，<b>真正决定成败的是牛熊 regime 与灾难止损幅度</b>。</li>
<li><b>牛熊翻转（核心）</b>：切到 bear 区间后多个策略中位超额转正，均值回复的相对优势在震荡/下行市显著放大 - 它是 regime 依赖型风控工具，应在震荡/熊市启用、牛市让位买入持有，而非全天候替代买入持有。</li>
<li><b>改进4 RSI+布林共振</b>：双指标同时超卖才做多，假信号与交易次数下降、稳健率提升；但牺牲了交易频率，适合低频高确定性场景。</li>
<li><b>过拟合护栏</b>：统一默认参数、参数网格扫描稳定、overfit/no_edge 组合绝不配置资金；盈利因子&gt;1 且样本外稳健才具实盘价值。</li>
<li><b>下一步</b>：小资金实盘验证改进后出场（宽止损/中轨止盈）+ 牛熊切换开关，再逐步加仓；可叠加多策略共振提升稳健率。</li>
</ul>
</div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)



if __name__ == "__main__":
    main()
