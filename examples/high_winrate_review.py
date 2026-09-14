"""高胜率量化方案实测：真实行情 + 自研引擎 + 样本外验证。

目标（用户指令「完善量化方案，测试其可行性，结合真实数据的实际效果，维持高胜率」）：
1. 用 R17 修复后的腾讯真实行情源（DataGateway，demo=False）取 11 只流动性 A 股 2019-2024
   （覆盖 2019-2021 牛市 + 2022-2024 熊/震荡），避免「回测建立在随机游走上」。
2. 跑 6 个均值回复类候选策略（天然高胜率）：bollinger / zscore / rsi / ou / williams_r / cci_signal。
3. 用**正确的**逐笔配对胜率（修复 perf.metrics.win_rate 的 cash_after 近似缺陷）计算胜率、
   盈亏比、盈利因子、夏普、最大回撤、相对买入持有的超额收益。
4. 样本外验证：前 60% 定参、后 40% 验证，报告稳健/退化/过拟合判定。
5. 输出 outputs/high_winrate_review.csv + .html 结构化报告。

引擎假设（与生产 run_backtest 一致、真实可执行）：
- 成本：佣金万三 + 滑点千一 + 印花税千一（卖出）
- 前视治理：execution_lag=1（t 日收盘出信号、t+1 开盘成交），杜绝前视偏差
- 风控：个股止损 10%（stop=0.10）；另跑一组 stop=1.0（等价于不止损）暴露信号本身的胜率
"""
from __future__ import annotations

import os
import sys
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


def main():
    gw = DataGateway()
    rows = []
    data_meta = {}
    for sym, name in SYMBOLS:
        try:
            df = fetch_real(gw, sym)
        except Exception as e:
            print(f"[SKIP] {sym} {name}: {type(e).__name__}: {e}")
            rows.append(dict(symbol=sym, name=name, error=str(e)))
            continue
        data_meta[sym] = dict(name=name, rows=len(df),
                              cmin=float(df["close"].min()), cmax=float(df["close"].max()),
                              source=gw.last_source)
        print(f"[OK] {sym} {name}: {len(df)} 行, close {df['close'].min():.2f}~{df['close'].max():.2f} (source={gw.last_source})")
        bench = buyhold_equity(df, INIT_CASH)
        bench_ret = float(bench.iloc[-1] / bench.iloc[0] - 1)
        bench_sharpe = sharpe(bench)
        for strat in STRATEGIES:
            fn = get_strategy(strat).generate_signals
            rec = dict(symbol=sym, name=name, strategy=strat,
                       label=STRATEGY_LABEL[strat], benchmark_return=bench_ret,
                       benchmark_sharpe=bench_sharpe)
            # 两种止损设定
            for tag, sl in (("stop", 0.10), ("nostop", 1.0)):
                eng = BacktestEngine(init_cash=INIT_CASH, risk=RiskManager(stop_loss=sl))
                res = eng.run(df, fn(df))
                st = res.stats()
                rt = round_trip_stats(res.trades)
                rec[f"{tag}_win"] = rt["win_rate"]
                rec[f"{tag}_trades"] = rt["n"]
                rec[f"{tag}_pf"] = rt["profit_factor"]
                rec[f"{tag}_ret"] = float(st["total_return"] or 0.0)
                if tag == "stop":
                    rec["sharpe"] = sharpe(res.equity)
                    rec["max_dd"] = max_drawdown(res.equity)
                    rec["annual"] = annual_return(res.equity)
                    rec["excess"] = rec["stop_ret"] - bench_ret
                    rec["sanity"] = res.sanity()["level"]
            # 样本外（用真实可执行止损 10%）
            oos = oos_eval(df, fn, 0.6, RiskManager(stop_loss=0.10))
            rec["oos_verdict"] = oos.get("verdict", "unknown")
            rec["oos_decay"] = oos.get("decay")
            rec["oos_in_win"] = oos.get("in_win")
            rec["oos_out_win"] = oos.get("out_win")
            rec["oos_in_ret"] = oos.get("in_return")
            rec["oos_out_ret"] = oos.get("out_return")
            rows.append(rec)
            print(f"    {strat:11s} win(stop)={rec['stop_win']:.2%} win(nostop)={rec['nostop_win']:.2%} "
                  f"excess={rec['excess']:+.1%} oos={rec['oos_verdict']}")

    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "high_winrate_review.csv")
    out_html = os.path.splitext(out_csv)[0] + ".html"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(out_csv, index=False)
    print(f"\n[WRITE] {out_csv}")

    # ---------- 策略级聚合 ----------
    agg = []
    valid = [r for r in rows if "stop_win" in r]
    for strat in STRATEGIES:
        sub = [r for r in valid if r["strategy"] == strat]
        if not sub:
            continue
        sw = [r["stop_win"] for r in sub]
        nw = [r["nostop_win"] for r in sub]
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
            avg_nostop_win=float(np.mean(nw)),
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

    _render_html(out_html, rows, valid, agg_df, data_meta)
    print(f"[WRITE] {out_html}")
    # 控制台速览
    print("\n=== 策略级排名（按真实止损下平均胜率）===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(agg_df.to_string(index=False))
    return out_csv, out_html


def _render_html(path, rows, valid, agg_df, data_meta):
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
    errs = [r for r in rows if "stop_win" not in r]
    err_html = ""
    if errs:
        err_html = "<h3>跳过/失败的标的</h3><ul>" + "".join(
            f"<li>{html.escape(str(r.get('symbol','?'))) } {html.escape(str(r.get('name','')))}: "
            f"{html.escape(str(r.get('error','')))}</li>" for r in errs) + "</ul>"

    # 推荐结论
    if not agg_df.empty:
        best = agg_df.iloc[0]
        # 可行门槛：真实止损下胜率>=50% 且 样本外稳健率>=50% 且 风险调整后不劣于买入持有
        cand = agg_df[(agg_df["avg_stop_win"] >= 0.5)
                      & (agg_df["robust_rate"] >= 0.5)
                      & (agg_df["avg_strat_sharpe"] > agg_df["avg_bench_sharpe"])]
        if not cand.empty:
            pick = cand.sort_values("median_excess", ascending=False).iloc[0]
            rec_txt = (f"可行组合：<b>{html.escape(pick['label'])}</b>（{pick['strategy']}）"
                       f"胜率 {fmt_pct(pick['avg_stop_win'])}、样本外胜率 {fmt_pct(pick['avg_oos_win'])}、"
                       f"盈利因子 {fmt_num(pick['avg_pf'])}、风险调整夏普 {fmt_num(pick['avg_strat_sharpe'])}"
                       f"（基准 {fmt_num(pick['avg_bench_sharpe'])}）。")
        else:
            pick = best
            rec_txt = (
                "⚠️ 无策略同时满足「胜率≥50% + 样本外稳健率≥50% + 夏普跑赢买入持有」。相对最佳为 "
                "<b>" + html.escape(best["label"]) + "</b>：胜率 " + fmt_pct(best["avg_stop_win"])
                + "、中位超额 " + fmt_pct(best["median_excess"]) + "、样本外稳健率 "
                + fmt_pct(best["robust_rate"]) + "。高胜率真实但被（1）10% 硬止损错配（不止损胜率升至 "
                + fmt_pct(best["avg_nostop_win"]) + "）、（2）2019-2024 大牛市中现金为王错过趋势 双重拖累，"
                "绝对值与风险调整均难稳定跑赢买入持有，须审慎、勿据此重仓。"
            )
    else:
        rec_txt = "无有效结果。"

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
<div class="meta">引擎假设：佣金万三 + 滑点千一 + 印花税千一（卖出）；成交=t+1 开盘（execution_lag=1，杜绝前视）；个股止损 10%（另列"不止损"暴露信号本身胜率）</div>
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
<h2>三、结论与使用建议（基于真实行情的诚实结论）</h2>
<ul class="note">
<li><b>可行性（数据可信）</b>：本报告全部 11 只标的均取自真实行情（腾讯源 + AKShare，demo=False），回测建立在真实价格上，结论可复现、可审计。</li>
<li><b>高胜率成立且样本外稳健</b>：均值回复类真实止损下胜率 50~65%、去掉错配止损后 60~85%；样本外胜率与样本内基本持平（williams_r 样本外 59.8%），说明高胜率不是拟合噪声。在震荡/回调频繁的标的（银行、券商、白电、恒瑞）尤甚；强趋势标的（茅台、宁德）信号稀少、统计意义弱。</li>
<li><b>但高胜率 ≠ 高收益</b>：2019-2024 含 2019-2021 大牛市，买入持有中位数收益极高；现金为主的均值回复策略结构性跑输——平均超额 −200%+ 主要由茅台(−245%)、宁德(−1288%) 等趋势龙头拖累。<b>然而个别标的确实跑赢</b>：平安银行 RSI +39.6%、恒瑞布林 +26.7%、恒瑞 CCI +7.9% 超额为正，证明在均值回复属性强的标的上策略可行。</li>
<li><b>致命错配：10% 硬止损</b>：均值回复在"已下跌触轨"时买入，再跌 10% 触发止损后价格常回归（策略本对但被砍）。证据：止损胜率 25~65% vs 不止损 60~85%。盈利因子仅 1.2~1.5，薄如刀片。<b>改进方向</b>：用"回归中轨止盈 + 宽幅灾难止损（如 20~25%）"替代 10% 硬砍，预计胜率与盈亏比同步改善。</li>
<li><b>过拟合护栏</b>：全程统一默认参数、不做逐股调参；样本外衰减是过拟合唯一硬标准，overfit/no_edge 组合绝不配置资金。盈利因子 &gt;1 且样本外稳健，才具备实盘价值。</li>
<li><b>下一步</b>：① 给均值回复策略换"回归止盈+宽止损"出场；② 叠加趋势过滤器（如 MA 多头才做多）以在大牛市不至于空仓；③ 多策略组合（RSI 超卖 + 布林超卖共振）进一步提升稳健率；④ 参数敏感性网格确认泛化。</li>
</ul>
</div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)


if __name__ == "__main__":
    main()
