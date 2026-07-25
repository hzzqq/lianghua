"""Lianghua Quant —— 多资产量化终端（Streamlit）。

启动：streamlit run lianghua/ui/app.py
支持资产：股票 / 基金 / 期货 / 期权，统一红涨绿跌配色。

页面导航（侧边栏）：
  单标的回测 / 组合·篮子回测 / 统一编排(多资产) / 期权组合策略 /
  期货跨期套利 / 基金筛选 / 风险度量VaR / 向量化回测 / 信号推送
"""
import sys
import tempfile
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOGO_PATH = ROOT / "assets" / "logo.png"

from lianghua.core.assets import AssetType
from lianghua.backtest.runner import run_backtest, SUPPORTED
from lianghua.strategy.registry import get_strategy as reg_get_strategy, STRATEGY_NAMES
from lianghua.portfolio.registry import get_optimizer as reg_get_optimizer, OPTIMIZER_NAMES
from lianghua.backtest.basket import basket_backtest
from lianghua.backtest.orchestrator import run_plan
from lianghua.backtest.vectorized import vectorized_backtest, positions_from_signals
from lianghua.backtest.future_spread import FutureSpread, spread_series
from lianghua.option.strategy import vertical_spread, iron_condor, butterfly, payoff_curve
from lianghua.fund.screener import FundScreener
from lianghua.risk.var import var_report
from lianghua.perf.metrics import report
from lianghua.execution.notify import (SignalNotifier, LogChannel, WebhookChannel,
                                      WeChatNotifier, WeComAppNotifier, NotifyHub,
                                      build_notifier)
from lianghua.backtest.montecarlo import bootstrap, parametric
from lianghua.optimize.param_search import grid_search, walk_forward
from lianghua.report import render_backtest_html
from lianghua.indicators import rsi, macd, kdj, boll
from lianghua.strategy.turtle import turtle_signals
from lianghua.strategy.donchian import donchian_signal
from lianghua.strategy.keltner import keltner_signal
from lianghua.strategy.ou_mean_reversion import ou_signal
from lianghua.strategy.dual_momentum import dual_momentum
from lianghua.strategy.rsi_strategy import rsi_signal
from lianghua.portfolio.clustering import cluster_assets
from lianghua.portfolio.optimizer import optimize
from lianghua.portfolio.min_variance import min_variance
from lianghua.portfolio.hrp import hrp
from lianghua.portfolio.black_litterman import black_litterman
from lianghua.factor.engine import FactorEngine
from lianghua.factor.returns import factor_long_short, factor_ic
from lianghua.factor.layer import ic_series
from lianghua.core.constants import ACCENT, COLOR_UP, COLOR_DOWN
from lianghua.indicators import INDICATOR_FUNCS
from lianghua.ui.widgets import safe_pct, safe_num, empty_figure
from lianghua.core.capabilities import list_capabilities, summary_counts
from lianghua.option.strategy import (OPTION_COMBO_REGISTRY, get_option_combo, payoff_curve)
from lianghua.data.sources import list_sources, fetch_from
from lianghua.data.universe import list_universes, get_universe
from lianghua.execution.brokers import make_broker
from lianghua.execution.orders import bracket_order, evaluate_bracket
import inspect, importlib, pkgutil

# 绩效 / 风险函数动态映射（扫描子模块，自动发现，含 extra2 / extra 等超目标新增）
def _load_mod_funcs(pkg_name: str) -> dict:
    out: dict = {}
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception:
        return out
    for _f, _m, _ispkg in pkgutil.iter_modules(pkg.__path__):
        if _m.startswith("__"):
            continue
        try:
            mod = importlib.import_module(f"{pkg_name}.{_m}")
        except Exception:
            continue
        for _n, _o in inspect.getmembers(mod, inspect.isfunction):
            if not _n.startswith("_"):
                out.setdefault(_n, _o)
    return out

PERF_FUNCS = _load_mod_funcs("lianghua.perf")
RISK_FUNCS = _load_mod_funcs("lianghua.risk")
from lianghua.data.gateway import DataGateway
from lianghua.execution.live import LiveEngine, make_live_engine
from lianghua.execution.order_book import OrderBook
from lianghua.execution.intraday import IntradayEngine, minute_signal
from lianghua.execution.router import make_multi_engine, AccountRouter
from lianghua.execution.schedule import is_trading_session, next_session

# A 股配色：红涨绿跌
RED = "#ff4d4f"
GREEN = "#00d486"
ACCENT = "#667eea"

ASSET_LABELS = {"stock": "股票", "fund": "基金", "future": "期货", "option": "期权"}

st.set_page_config(page_title="Lianghua Quant 多资产终端", page_icon="📈", layout="wide")
st.title("📈 Lianghua Quant · 多资产量化交易终端")

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=140)
    st.divider()
    PAGE = st.radio("功能导航", [
        "单标的回测", "组合·篮子回测", "统一编排(多资产)",
        "期权组合策略", "期货跨期套利", "基金筛选",
        "风险度量 VaR", "向量化回测", "蒙特卡洛模拟",
        "参数优化", "HTML报告", "信号推送",
        "策略库", "组合优化", "因子研究",
        "指标实验室", "绩效风险分析", "数据执行能力", "能力总览",
        "实盘交易", "多账户与定时",
    ])


# ---------------- 通用辅助 ----------------
def equity_chart(equity: pd.Series, title: str = "组合净值"):
    eq = pd.Series(equity).astype(float) if equity is not None else pd.Series(dtype=float)
    if len(eq) == 0 or not np.isfinite(eq).any():
        return empty_figure("无有效净值数据")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="净值",
                             line=dict(color=ACCENT)))
    fig.update_layout(height=420, template="plotly_dark",
                      margin=dict(l=20, r=20, t=30, b=20), title=title)
    return fig


def metric_row(m: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("期末权益", safe_num(m.get("final_equity"), 0))
    c2.metric("总收益", safe_pct(m.get("total_return")))
    c3.metric("年化收益", safe_pct(m.get("annual_return")))
    c4.metric("夏普比率", safe_num(m.get("sharpe")))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("最大回撤", safe_pct(m.get("max_drawdown")))
    c6.metric("交易次数", int(m.get("num_trades", 0) or 0))
    c7.metric("胜率", safe_pct(m.get("win_rate")))
    c8.metric("超额收益", safe_pct(m.get("excess_return")))


# ---------------- 1. 单标的回测 ----------------
def page_single():
    st.header("单标的回测")
    with st.sidebar:
        pass
    asset = st.selectbox("资产类别", list(ASSET_LABELS.keys()),
                         format_func=lambda x: ASSET_LABELS[x], key="s_asset")
    hints = {
        "stock": "如 600519.SH / 000001.SZ",
        "fund": "如 110011.OF（开放式）/ 510300.SH（ETF）",
        "future": "如 RB2410.SHF（螺纹）/ IF2409.CFE（股指）/ RB0.SHF（主连）",
        "option": "如 510050C3000.SH（50ETF购）/ 510050P3000.SH（沽）",
    }
    symbol = st.text_input("标的代码", value="600519.SH", help=hints[asset], key="s_symbol")
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("开始日期", value=pd.to_datetime("2023-01-01"), key="s_start")
    with c2:
        end = st.date_input("结束日期", value=pd.to_datetime("2024-12-31"), key="s_end")
    strategy = st.selectbox("策略", SUPPORTED[asset], index=0, key="s_strat")
    init_cash = st.number_input("初始资金(元)", value=1_000_000, step=100_000, key="s_cash")
    stop_loss = st.number_input("止损线", value=0.10, format="%.2f", key="s_sl")
    run = st.button("▶ 运行回测", type="primary", key="s_run")

    if run:
        with st.spinner("拉取数据并回测中..."):
            from lianghua.risk.manager import RiskManager
            res = run_backtest(symbol, str(start), str(end), strategy=strategy,
                               asset=asset, init_cash=init_cash,
                               risk=RiskManager(stop_loss=stop_loss))
            perf = report(res.equity, res.trades, df=res.df, init_cash=init_cash)
        metric_row(perf)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity.values,
                                 name="净值", line=dict(color=ACCENT)))
        fig.update_layout(height=420, template="plotly_dark",
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        if res.trades:
            st.subheader("成交明细")
            st.dataframe(pd.DataFrame(res.trades), use_container_width=True)


# ---------------- 2. 组合·篮子回测 ----------------
def page_basket():
    st.header("组合 · 篮子回测")
    st.caption("输入一篮子标的与权重（每行：代码,权重），按归一化价格做加权组合回测。")
    default = "600519.SH,0.4\n510300.SH,0.3\n000300.SH,0.3"
    txt = st.text_area("标的与权重", value=default, height=120)
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("开始", value=pd.to_datetime("2023-01-01"), key="b_start")
    with c2:
        end = st.date_input("结束", value=pd.to_datetime("2023-12-31"), key="b_end")
    with c3:
        init_cash = st.number_input("初始资金", value=1_000_000, step=100_000, key="b_cash")
    if st.button("▶ 运行篮子回测", type="primary", key="b_run"):
        specs = []
        for line in txt.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                specs.append({"symbol": parts[0], "weight": float(parts[1])})
        if not specs:
            st.error("请至少输入一个 代码,权重")
            return
        with st.spinner("回测中..."):
            r = basket_backtest(specs, str(start), str(end), init_cash=init_cash)
        metric_row(r["metrics"])
        st.plotly_chart(equity_chart(r["equity"], "篮子组合净值"), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("权重")
            st.dataframe(pd.DataFrame({"标的": list(r["weights"]),
                                       "权重": list(r["weights"].values())}),
                         use_container_width=True)
        with col2:
            st.subheader("收益归因(贡献)")
            st.dataframe(pd.DataFrame({"标的": list(r["attribution"]),
                                       "贡献": list(r["attribution"].values())}),
                         use_container_width=True)


# ---------------- 3. 统一编排(多资产) ----------------
def page_orchestrator():
    st.header("统一编排 · 多资产配置方案")
    st.caption("一键编排『多资产 + 多策略』组合：逐标的回测 → 按权重合并净值 → 汇总绩效。")
    default = ("600519.SH,stock,sma_cross,0.4\n"
               "510300.SH,fund,momentum,0.4\n"
               "RB0.SHF,future,breakout,0.2")
    txt = st.text_area("计划(代码,资产,策略,权重)", value=default, height=130)
    c1, c2 = st.columns(2)
    with c1:
        init_cash = st.number_input("初始资金", value=1_000_000, step=100_000, key="o_cash")
    with c2:
        st.info("资产可选: stock/fund/future/option")
    if st.button("▶ 一键编排", type="primary", key="o_run"):
        plan = []
        for line in txt.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                plan.append({"symbol": parts[0], "asset": parts[1],
                             "strategy": parts[2], "weight": float(parts[3])})
        if not plan:
            st.error("请至少输入一行 代码,资产,策略,权重")
            return
        with st.spinner("逐标的回测并合并..."):
            r = run_plan(plan, init_cash=init_cash)
        metric_row(r["metrics"])
        st.plotly_chart(equity_chart(r["equity"], "组合净值(多资产编排)"), use_container_width=True)
        st.subheader("成分明细")
        rows = []
        for sym, c in r["components"].items():
            rows.append({"标的": sym, "资产": c["asset"], "策略": c["strategy"],
                         "期末权益": round(c["metrics"]["final_equity"], 0),
                         "夏普": round(c["metrics"]["sharpe"], 2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------- 4. 期权组合策略 ----------------
def page_option():
    st.header("期权组合策略 · 损益曲线")
    st.caption("共 %d 种组合（注册表驱动，含迭代181+新增）：备兑/保护/领口/跨式/宽跨/铁鹰/铁蝶/日历/比率/对角/垂直/单腿" % len(OPTION_COMBO_REGISTRY))
    combo = st.selectbox("组合类型", list(OPTION_COMBO_REGISTRY.keys()))
    center = st.number_input("标的中心价（重建组合参考）", value=100.0, key="op_center")
    s_lo = st.number_input("标的下界", value=80.0, key="op_slo")
    s_hi = st.number_input("标的上界", value=120.0, key="op_shi")
    if st.button("▶ 计算损益曲线", type="primary", key="op_run"):
        # 以中心价为锚重建组合（沿用各组合默认间距），保证一键可预览
        spec = OPTION_COMBO_REGISTRY[combo]
        kw = dict(spec["defaults"])
        # 把含 strike 语义的参数平移到 center 附近
        for kk in ("spot", "strike", "low", "mid", "high", "body", "wing",
                  "near_k", "far_k", "call_k", "put_k", "entry"):
            if kk in kw and isinstance(kw[kk], (int, float)):
                kw[kk] = round(float(kw[kk]) * center / 100.0, 2)
        legs = get_option_combo(combo, **kw)
        pc = payoff_curve(legs, s_lo, s_hi)
        title = f"{combo} · {spec['desc']}"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pc["S"], y=pc["pnl"], name="到期损益",
                                 line=dict(color=ACCENT), fill="tozeroy"))
        fig.add_hline(y=0, line=dict(color="#888", dash="dash"))
        fig.update_layout(height=420, template="plotly_dark", title=title,
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"最大盈利≈{pc['pnl'].max():.2f}｜最大亏损≈{pc['pnl'].min():.2f}｜"
                   f"盈亏平衡≈{pc.loc[pc['pnl'].abs().idxmin(),'S']:.2f}")
        st.subheader("组合腿")
        st.dataframe(pd.DataFrame([{"腿": i + 1, "方向": l.side.upper(),
                                    "类型": l.otype, "行权价": l.strike,
                                    "权利金": l.premium}
                                   for i, l in enumerate(legs)]),
                     use_container_width=True)


# ---------------- 5. 期货跨期套利 ----------------
def page_future_spread():
    st.header("期货跨期套利 · 价差均值回归")
    c1, c2 = st.columns(2)
    with c1:
        near = st.text_input("近月合约", value="RB2410.SHF", key="fs_near")
        far = st.text_input("远月合约", value="RB2501.SHF", key="fs_far")
    with c2:
        start = st.date_input("开始", value=pd.to_datetime("2023-01-01"), key="fs_start")
        end = st.date_input("结束", value=pd.to_datetime("2023-12-31"), key="fs_end")
    c3, c4, c5, c6 = st.columns(4)
    with c3:
        window = st.number_input("窗口", value=20, key="fs_w")
    with c4:
        entry_z = st.number_input("入场z", value=1.5, key="fs_ez")
    with c5:
        exit_z = st.number_input("退出z", value=0.5, key="fs_xz")
    with c6:
        lots = st.number_input("手数", value=2, key="fs_lots")
    multiplier = st.number_input("合约乘数", value=10.0, key="fs_mult")
    if st.button("▶ 运行套利回测", type="primary", key="fs_run"):
        from lianghua.data.gateway import DataGateway
        gw = DataGateway()
        with st.spinner("拉取近/远月数据..."):
            near_df = gw.fetch(near, str(start), str(end), asset=AssetType.FUTURE)
            far_df = gw.fetch(far, str(start), str(end), asset=AssetType.FUTURE)
        if near_df.empty or far_df.empty:
            st.error("近月或远月数据为空")
            return
        fs = FutureSpread(window=int(window), entry_z=entry_z, exit_z=exit_z,
                          lots=int(lots), multiplier=multiplier)
        r = fs.run(near_df, far_df)
        metric_row(r["metrics"])
        fig = make_2row(r["spread"], r["equity"], "价差", "净值")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"信号次数={(r['signals'] != 0).sum()}｜交易次数={len(r['trades'])}")


# ---------------- 6. 基金筛选 ----------------
def page_fund():
    st.header("基金多因子筛选")
    st.caption("输入若干基金代码（每行一个），自动拉取净值做动量/波动/回撤/夏普加权打分排名。")
    default = "510300.SH\n110011.OF\n161725.OF"
    txt = st.text_area("基金代码列表", value=default, height=100)
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("开始", value=pd.to_datetime("2022-01-01"), key="fd_start")
    with c2:
        end = st.date_input("结束", value=pd.to_datetime("2023-12-31"), key="fd_end")
    with c3:
        topn = st.number_input("展示TOP", value=10, key="fd_topn")
    if st.button("▶ 开始筛选", type="primary", key="fd_run"):
        from lianghua.data.gateway import DataGateway
        gw = DataGateway()
        codes = [x.strip() for x in txt.strip().splitlines() if x.strip()]
        navs = {}
        with st.spinner("拉取基金净值..."):
            for code in codes:
                df = gw.fetch(code, str(start), str(end), asset=AssetType.FUND)
                if not df.empty:
                    navs[code] = df.set_index("date")["close"].astype(float)
        if not navs:
            st.error("未能获取任何基金净值")
            return
        frame = pd.DataFrame(navs)
        sc = FundScreener.from_frame(frame)
        res = sc.screen().head(int(topn))
        st.dataframe(res, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=res.index.astype(str), y=res["综合分"],
                             marker_color=ACCENT))
        fig.update_layout(height=360, template="plotly_dark", title="综合得分排名",
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ---------------- 7. 风险度量 VaR ----------------
def page_var():
    st.header("风险度量 · VaR / CVaR")
    st.caption("先对单标的回测得到收益序列，再计算历史/参数 VaR 与 CVaR（期望损失）。")
    symbol = st.text_input("标的代码", value="600519.SH", key="v_symbol")
    strategy = st.selectbox("策略", SUPPORTED["stock"], index=0, key="v_strat")
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("开始", value=pd.to_datetime("2023-01-01"), key="v_start")
    with c2:
        end = st.date_input("结束", value=pd.to_datetime("2023-12-31"), key="v_end")
    init_cash = st.number_input("名义本金", value=1_000_000, step=100_000, key="v_cash")
    if st.button("▶ 计算风险", type="primary", key="v_run"):
        res = run_backtest(symbol, str(start), str(end), strategy=strategy,
                           asset="stock", init_cash=init_cash)
        rets = res.equity.pct_change().dropna()
        rep = var_report(rets, confs=(0.90, 0.95, 0.99), notional=init_cash)
        st.dataframe(rep, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=rets.values, nbinsx=60, marker_color=ACCENT))
        fig.update_layout(height=360, template="plotly_dark", title="日收益率分布",
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ---------------- 8. 向量化回测 ----------------
def page_vectorized():
    st.header("向量化回测（加速）")
    st.caption("纯 pandas 向量化信号→持仓→净值，适合大样本快速扫描。")
    symbol = st.text_input("标的代码", value="600519.SH", key="vb_symbol")
    strategy = st.selectbox("策略", STRATEGY_NAMES, index=0, key="vb_strat")
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("开始", value=pd.to_datetime("2023-01-01"), key="vb_start")
    with c2:
        end = st.date_input("结束", value=pd.to_datetime("2023-12-31"), key="vb_end")
    with c3:
        cost_rate = st.number_input("换手成本", value=0.0005, format="%.4f", key="vb_cost")
    init_cash = st.number_input("初始资金", value=1_000_000, step=100_000, key="vb_cash")
    if st.button("▶ 运行向量化回测", type="primary", key="vb_run"):
        from lianghua.data.gateway import DataGateway
        gw = DataGateway()
        df = gw.fetch(symbol, str(start), str(end), asset=AssetType.STOCK)
        sig = reg_get_strategy(strategy).generate_signals(df)
        r = vectorized_backtest(df, sig, init_cash=init_cash, cost_rate=cost_rate)
        metric_row(r["metrics"])
        st.metric("换手次数", r["num_changes"])
        fig = make_2row(r["equity"], r["positions"], "净值", "持仓", bar2=False)
        st.plotly_chart(fig, use_container_width=True)


# ---------------- 9. 信号推送 ----------------
def page_notify():
    st.header("实盘信号推送")
    st.caption("把交易信号推送到本地日志（可选 Webhook）。对齐 Broker 接口风格。")
    symbol = st.text_input("标的代码", value="600519.SH", key="n_symbol")
    action = st.selectbox("动作", ["BUY", "SELL", "HOLD"], key="n_action")
    price = st.number_input("价格", value=0.0, key="n_price")
    strategy = st.text_input("策略名", value="sma_cross", key="n_strat")
    webhook = st.text_input("Webhook URL（可选）", value="", key="n_web")
    log_file = st.text_input("日志文件路径（可选）", value="", key="n_log")
    if st.button("▶ 发送测试信号", type="primary", key="n_run"):
        channels = []
        if log_file:
            channels.append(LogChannel(log_file))
        if webhook:
            channels.append(WebhookChannel(webhook))
        notifier = SignalNotifier(channels=channels if channels else None)
        result = notifier.notify_signal(symbol, action, price=price, strategy=strategy)
        st.success(f"已推送：{result['delivered']}/{result['total']} 渠道成功")
        st.json(result["results"])
        st.subheader("推送历史")
        st.dataframe(pd.DataFrame([h["event"] for h in notifier.history]),
                     use_container_width=True)


# ---------------- 10. 蒙特卡洛模拟 ----------------
def page_montecarlo():
    st.header("蒙特卡洛稳健性模拟")
    st.caption("对历史回测权益曲线做重采样，评估策略收益的尾部分布与破产概率。")
    symbol = st.text_input("标的代码", value="600519.SH", key="mc_symbol")
    strategy = st.selectbox("策略", SUPPORTED["stock"], index=0, key="mc_strat")
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("开始", value=pd.to_datetime("2023-01-01"), key="mc_start")
    with c2:
        end = st.date_input("结束", value=pd.to_datetime("2023-12-31"), key="mc_end")
    with c3:
        init_cash = st.number_input("初始资金", value=1_000_000, step=100_000, key="mc_cash")
    c4, c5, c6 = st.columns(3)
    with c4:
        method = st.selectbox("方法", ["bootstrap", "parametric"], key="mc_method")
    with c5:
        n_sim = st.number_input("模拟次数", value=1000, step=100, key="mc_n")
    with c6:
        block = st.number_input("块长度", value=5, key="mc_block")
    if st.button("▶ 运行蒙特卡洛", type="primary", key="mc_run"):
        res = run_backtest(symbol, str(start), str(end), strategy=strategy,
                           asset="stock", init_cash=init_cash)
        eq = res.equity
        with st.spinner("重采样中..."):
            mc = (bootstrap(eq, n=int(n_sim), block=int(block), init_cash=init_cash)
                  if method == "bootstrap"
                  else parametric(eq, n=int(n_sim), init_cash=init_cash))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("中位收益", f"{mc['median_return'] * 100:.1f}%")
        c2.metric("P5~P95", f"{mc['p5'] * 100:.1f}% ~ {mc['p95'] * 100:.1f}%")
        c3.metric("亏损概率", f"{mc['prob_loss'] * 100:.1f}%")
        c4.metric("最差回撤", f"{mc['worst_max_drawdown'] * 100:.1f}%")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=mc["returns"] * 100, nbinsx=60, marker_color=ACCENT))
        fig.update_layout(height=380, template="plotly_dark",
                          title="模拟期末收益分布(%)", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ---------------- 11. 参数优化 ----------------
def page_param():
    st.header("参数优化 · 网格 / Walk-forward")
    st.caption("以 SMA 双均线为例，遍历快慢窗口组合寻找更优参数；walk-forward 评估样本外稳健性。")
    symbol = st.text_input("标的代码", value="600519.SH", key="pa_symbol")
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("开始", value=pd.to_datetime("2023-01-01"), key="pa_start")
    with c2:
        end = st.date_input("结束", value=pd.to_datetime("2023-12-31"), key="pa_end")
    fast_range = st.text_input("快线窗口(逗号)", value="5,10,15,20", key="pa_fast")
    slow_range = st.text_input("慢线窗口(逗号)", value="30,40,50,60", key="pa_slow")
    mode = st.selectbox("模式", ["grid", "walk_forward"], key="pa_mode")
    init_cash = st.number_input("初始资金", value=1_000_000, step=100_000, key="pa_cash")

    def make_sma_signals(df, fast, slow):
        c = df["close"].astype(float)
        return (c.rolling(int(fast)).mean() > c.rolling(int(slow)).mean()).astype(int)

    def rb(df, sig):
        return vectorized_backtest(df, sig, init_cash=init_cash)["equity"]

    if st.button("▶ 开始优化", type="primary", key="pa_run"):
        from lianghua.data.gateway import DataGateway
        gw = DataGateway()
        df = gw.fetch(symbol, str(start), str(end), asset=AssetType.STOCK)
        if df.empty:
            st.error("数据为空")
            return
        fast_l = [int(x) for x in fast_range.split(",") if x.strip()]
        slow_l = [int(x) for x in slow_range.split(",") if x.strip()]
        grid = {"fast": fast_l, "slow": slow_l}
        with st.spinner("遍历参数中..."):
            if mode == "grid":
                res = grid_search(df, make_sma_signals, grid, run_backtest=rb)
                st.subheader("参数评分排名（按夏普降序）")
                st.dataframe(res, use_container_width=True)
            else:
                oos, best = walk_forward(df, make_sma_signals, grid, run_backtest=rb)
                oos_mean = float(np.nanmean(oos)) if oos else float("nan")
                c1, c2 = st.columns(2)
                c1.metric("平均样本外夏普", f"{oos_mean:.3f}")
                c2.metric("窗口数", len(oos))
                st.subheader("各窗口最优参数")
                st.dataframe(pd.DataFrame(best), use_container_width=True)


# ---------------- 12. HTML 报告 ----------------
def page_html():
    st.header("回测 HTML 报告")
    st.caption("生成单文件 HTML 回测报告（内联 SVG 曲线 + 指标卡 + 交易表 + 可选蒙特卡洛），可预览并下载。")
    symbol = st.text_input("标的代码", value="600519.SH", key="h_symbol")
    strategy = st.selectbox("策略", SUPPORTED["stock"], index=0, key="h_strat")
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("开始", value=pd.to_datetime("2023-01-01"), key="h_start")
    with c2:
        end = st.date_input("结束", value=pd.to_datetime("2023-12-31"), key="h_end")
    init_cash = st.number_input("初始资金", value=1_000_000, step=100_000, key="h_cash")
    with_mc = st.checkbox("含蒙特卡洛章节", value=True, key="h_mc")
    if st.button("▶ 生成报告", type="primary", key="h_run"):
        res = run_backtest(symbol, str(start), str(end), strategy=strategy,
                           asset="stock", init_cash=init_cash)
        html_str = render_backtest_html(res, symbol, strategy, "stock", init_cash, with_mc=with_mc)
        st.components.v1.html(html_str, height=900, scrolling=True)
        out_dir = ROOT / "reports"
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / f"backtest_{symbol}_{start}_{end}_{strategy}.html"
        fname.write_text(html_str, encoding="utf-8")
        st.success(f"报告已生成：{fname}")
        st.download_button("⬇ 下载 HTML", html_str, file_name=fname.name, mime="text/html")


# ---------------- 两行图辅助 ----------------
def make_2row(s1, s2, name1, name2, bar2=True):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=s1.index, y=s1.values, name=name1,
                             line=dict(color=ACCENT)), row=1, col=1)
    if bar2:
        fig.add_trace(go.Bar(x=s2.index, y=s2.values, name=name2,
                             marker_color="#8898ff"), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=s2.index, y=s2.values, name=name2,
                                 line=dict(color=GREEN)), row=2, col=1)
    fig.update_layout(height=520, template="plotly_dark",
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig


# ---------------- 策略库（迭代31-100） ----------------
def page_strategies():
    st.header("📈 策略库")
    st.caption("迭代 31–100 全部单标的择时/技术指标策略（统一注册表驱动，共 %d 个）" % len(STRATEGY_NAMES))
    c1, c2 = st.columns(2)
    with c1:
        symbol = st.text_input("标的代码", "600519.SH", key="sl_symbol")
    with c2:
        strat = st.selectbox("策略", STRATEGY_NAMES, index=0, key="sl_strat")
    start = str(st.date_input("开始", datetime.date(2024, 1, 1), key="sl_start"))
    end = str(st.date_input("结束", datetime.date(2024, 6, 30), key="sl_end"))
    if st.button("生成信号", key="sl_run"):
        try:
            gw = DataGateway()
            df = gw.fetch(symbol, start, end, asset=AssetType.STOCK)
            if df.empty:
                st.error("无数据（离线降级演示）")
                return
            sig = reg_get_strategy(strat).generate_signals(df)
            res = vectorized_backtest(df, sig)
            st.plotly_chart(equity_chart(res["equity"], f"{symbol} · {strat}"))
            m = res["metrics"]
            cc1, cc2, cc3, cc4 = st.columns(4)
            eq_first = res["equity"].iloc[0]
            cum = (res["equity"].iloc[-1] / eq_first - 1) if (pd.notna(eq_first) and eq_first != 0) else float("nan")
            cc1.metric("累计%", safe_pct(cum))
            cc2.metric("夏普", safe_num(m["sharpe"]))
            cc3.metric("回撤%", safe_pct(m["max_drawdown"]))
            cc4.metric("换手", f"{res['num_changes']}")
        except Exception as e:
            st.error(f"策略运行失败：{e}")


# ---------------- 组合优化（迭代39-50/85-100） ----------------
def page_portfolio():
    st.header("🧮 组合优化")
    st.caption("统一优化器注册表驱动（共 %d 种）+ 聚类；生成合成收益矩阵后一键求解权重" % len(OPTIMIZER_NAMES))
    method = st.selectbox("方法", OPTIMIZER_NAMES + ["clustering"])
    n = st.slider("资产数", 2, 6, 4)
    seed = st.number_input("随机种子", 0, 999, 0)
    if st.button("计算权重", key="pf_run"):
        try:
            rng = np.random.default_rng(int(seed))
            cols = [f"A{i}" for i in range(int(n))]
            rets = pd.DataFrame(rng.normal(0.0005, 0.01, (250, int(n))), columns=cols)
            if method == "clustering":
                cl = cluster_assets(rets, min(2, int(n)))
                st.json({f"簇{k}": v for k, v in cl.items()})
                w = pd.Series(1.0 / int(n), index=cols)
            else:
                w = reg_get_optimizer(method)(rets)
            wser = pd.Series(w, index=cols)
            st.dataframe(wser.rename("权重").to_frame())
            fig = go.Figure(go.Bar(x=cols, y=wser.values, marker_color=ACCENT))
            fig.update_layout(template="plotly_dark", height=300, title="权重分配")
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"优化失败：{e}")


# ---------------- 因子研究（迭代15/97） ----------------
def page_factor():
    st.header("🔬 因子研究")
    st.caption("内置因子 → IC / 多空收益（演示随机因子）")
    symbol = st.text_input("标的代码（取真实/演示价格序列）", "600519.SH")
    start = str(st.date_input("开始", datetime.date(2023, 1, 1)))
    end = str(st.date_input("结束", datetime.date(2023, 12, 31)))
    if st.button("运行因子研究"):
        try:
            gw = DataGateway()
            df = gw.fetch(symbol, start, end, asset=AssetType.STOCK)
            close = df.set_index("date")["close"]
            # 演示：用内置因子引擎 + 随机因子
            fe = FactorEngine(df)
            fwd = close.pct_change(5).shift(-5)
            rng = np.random.default_rng(0)
            rand_f = pd.Series(rng.normal(0, 1, len(close)), index=close.index)
            ls = factor_long_short(rand_f, fwd)
            ic = factor_ic(rand_f, fwd)
            mom = fe.compute("momentum_20")
            ev = fe.evaluate(mom, forward=5)
            icv = ic_series(mom, close.pct_change(5).shift(-5), lags=10)
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("随机因子IC", safe_num(ic, 3))
            cc2.metric("动量IC", safe_num(ev.get("ic", float("nan")), 3))
            cc3.metric("多空收益", safe_num(ls.iloc[0] if len(ls) else float("nan"), 4))
            st.line_chart(pd.Series(icv, name="动量IC衰减"))
        except Exception as e:
            st.error(f"因子研究失败：{e}")


# ---------------- 指标实验室（迭代181+，超目标） ----------------
def page_indicator_lab():
    st.header("🧪 指标实验室")
    st.caption("动态读取全部技术指标（共 %d 个，注册表驱动），选一个在演示行情上运行并绘图。" % len(INDICATOR_FUNCS))
    name = st.selectbox("指标", sorted(INDICATOR_FUNCS.keys()))
    symbol = st.text_input("标的代码", "600519.SH", key="il_sym")
    start = str(st.date_input("开始", datetime.date(2023, 1, 1), key="il_start"))
    end = str(st.date_input("结束", datetime.date(2023, 12, 31), key="il_end"))
    if st.button("▶ 运行指标", key="il_run"):
        try:
            df = DataGateway().fetch(symbol, start, end, asset=AssetType.STOCK)
            if df is None or df.empty:
                st.error("无数据（离线降级演示）")
                return
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            fn = INDICATOR_FUNCS[name]
            try:
                out = fn(df)
            except Exception:
                out = fn(df["close"])
            if isinstance(out, pd.Series):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=out.values, name=name,
                                         line=dict(color=ACCENT)))
                fig.update_layout(height=380, template="plotly_dark", title=name,
                                  margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(out, use_container_width=True)
            st.caption(f"输出维度：{getattr(out, 'shape', 'scalar')}")
        except Exception as e:
            st.error(f"指标运行失败：{e}")


# ---------------- 绩效·风险分析（迭代181+，超目标） ----------------
def page_perf_risk():
    st.header("📊 绩效 · 风险分析")
    st.caption("动态读取绩效(%d)/风险(%d)函数，选一个在演示净值上运行，查看标量结果。"
               % (len(PERF_FUNCS), len(RISK_FUNCS)))
    tab1, tab2 = st.tabs(["绩效函数", "风险函数"])
    with tab1:
        pname = st.selectbox("绩效函数", sorted(PERF_FUNCS.keys()))
    with tab2:
        rname = st.selectbox("风险函数", sorted(RISK_FUNCS.keys()))
    symbol = st.text_input("标的代码", "600519.SH", key="pr_sym")
    start = str(st.date_input("开始", datetime.date(2023, 1, 1), key="pr_start"))
    end = str(st.date_input("结束", datetime.date(2023, 12, 31), key="pr_end"))
    if st.button("▶ 运行", key="pr_run"):
        try:
            df = DataGateway().fetch(symbol, start, end, asset=AssetType.STOCK)
            if df is None or df.empty:
                st.error("无数据（离线降级演示）")
                return
            eq = df.set_index("date")["close"].astype(float)
            for label, funcs, nm in (("绩效", PERF_FUNCS, pname), ("风险", RISK_FUNCS, rname)):
                fn = funcs[nm]
                try:
                    val = fn(eq)
                except TypeError:
                    try:
                        val = fn(eq, eq)
                    except Exception:
                        val = fn(eq, eq.pct_change().fillna(0))
                except Exception as e:
                    st.error(f"{label}函数 {nm} 需额外参数，可在代码中直接调用：{e}")
                    continue
                if isinstance(val, (int, float, np.floating, np.integer)):
                    st.metric(f"{label} · {nm}", f"{float(val):.4f}")
                else:
                    st.write(f"{label} · {nm}：", val)
        except Exception as e:
            st.error(f"运行失败：{e}")


# ---------------- 数据·执行能力（迭代181+，超目标） ----------------
def page_data_exec():
    st.header("🔌 数据 · 执行能力")
    st.subheader("数据源（多源适配器）")
    st.write("可用源：", list_sources())
    src = st.selectbox("选择数据源", list_sources())
    sym = st.text_input("标的代码", "600519.SH", key="de_sym")
    if st.button("▶ 取数预览", key="de_fetch"):
        df = fetch_from(src, sym, "2023-01-01", "2023-06-30", asset=AssetType.STOCK)
        if df is None or df.empty:
            st.error("该源无数据（离线或库缺失，可换 synthetic）")
        else:
            st.dataframe(df.head(), use_container_width=True)
    st.subheader("成分 universe")
    univ = st.selectbox("universe", list_universes())
    if st.button("▶ 查看成分", key="de_univ"):
        st.json(get_universe(univ))
    st.subheader("纸面券商 + 括号单演示")
    if st.button("▶ 运行执行演示", key="de_run"):
        broker = make_broker("paper", slippage=0.001)
        o = bracket_order(sym, "BUY", 100, 100.0, 95.0, 108.0, asset_type=AssetType.STOCK)
        res = broker.submit(o["entry"])
        acct = broker.get_account({sym: 100.0 * 100})
        fired = evaluate_bracket(o, 101.0)
        st.json({"成交": res, "账户": acct, "触发腿": fired})


# ---------------- 能力总览（迭代181+） ----------------
def page_capabilities():
    st.header("🗺️ 能力总览")
    cap = list_capabilities()
    sc = summary_counts()
    st.subheader("平台能力计数")
    st.json(sc)
    c1, c2 = st.columns(2)
    with c1:
        st.write("**指标数**：", len(cap["indicators"]))
        st.write("**期权组合策略**：", list(cap["option_combos"].keys()))
    with c2:
        st.write("**数据源**：", cap["data"])
        st.write("**执行能力**：", cap["execution"])
    st.subheader("策略 / 优化器（注册表驱动，自动发现）")
    st.write("**策略**：", STRATEGY_NAMES)
    st.write("**优化器**：", OPTIMIZER_NAMES)


# ---------------- 路由 ----------------


# ---------------- 实盘交易（连接实盘） ----------------
def page_live():
    """实盘/纸面交易控制台：broker 选择 + 策略 + 标的 + 资金/风控，启动/停止/紧急停止，持仓与订单快照。

    默认 paper 模式（真实行情网关 + 模拟成交，零资金风险）；选择 qmt/pt 并勾选
    "确认连接真实柜台"后才会进入真实下单路径（connect 本身需 SDK + 账户就绪）。
    """
    st.header("实盘交易（连接实盘）")
    st.caption("把策略信号经盘前风控后路由到 broker；paper=模拟成交，qmt/pt=真实柜台（需 SDK+账户）。")

    broker_kind = st.selectbox("Broker 类型", ["paper", "sim", "qmt", "pt"])
    strategy = st.selectbox("策略", STRATEGY_NAMES)
    symbols = st.text_input("标的（逗号分隔）", "600519.SH,000300.SH")
    capital = st.number_input("资金（元）", min_value=10000, value=1000000, step=10000)
    lookback = st.number_input("回看交易日", min_value=20, value=120, step=10)
    max_order_value = st.number_input("单笔上限（元，0=不限）", min_value=0, value=200000, step=10000)
    confirm_live = st.checkbox("确认连接真实柜台（qmt/pt 生效；否则拒绝真实下单）")
    live = confirm_live and broker_kind in ("qmt", "pt")

    col1, col2, col3 = st.columns(3)
    if col1.button("启动一次调仓"):
        try:
            syms = [s.strip() for s in symbols.split(",") if s.strip()]
            eng = make_live_engine(
                broker_kind, strategy, syms, capital=capital,
                lookback=int(lookback), live=live,
                risk_limits={"max_order_value": float(max_order_value) or float("inf")},
            )
            st.session_state["live_engine"] = eng
            status = eng.step()
            st.success("已执行（mode=%s，动作数=%d）" % (status["mode"], len(status.get("actions", []))))
            _render_live_status(status)
        except Exception as e:
            st.error("启动失败：" + str(e))
    if col2.button("紧急停止"):
        eng = st.session_state.get("live_engine")
        if eng is not None:
            eng.kill()
            st.warning("已发送紧急停止信号（后续 step 将直接跳过）")
        else:
            st.info("当前无运行中的引擎")
    if col3.button("刷新订单账本"):
        book = OrderBook()
        rows = book.recent(30)
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("账本为空")


def _render_live_status(status: dict):
    """渲染一次调仓状态快照。"""
    acct = status.get("account") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("现金", "%.0f" % acct.get("cash", 0))
    c2.metric("持仓市值", "%.0f" % acct.get("position_value", 0))
    c3.metric("总权益", "%.0f" % acct.get("equity", 0))
    actions = status.get("actions", [])
    if actions:
        st.subheader("本次调仓动作")
        st.dataframe(pd.DataFrame(actions))
    orders = status.get("orders", [])
    if orders:
        st.subheader("最近订单")
        st.dataframe(pd.DataFrame(orders))

# ---------------- 多账户与定时调仓（第十三轮深化） ----------------
_MULTI_JSON = '''{
  "symbols": ["600519.SH", "IF.CFE"],
  "routes": {"by_asset": {"stock": "A", "future": "B"}, "default": "A"},
  "accounts": [
    {"name": "A", "broker": "paper", "strategy": "sma_cross", "capital": 1000000.0},
    {"name": "B", "broker": "paper", "strategy": "macd_cross", "capital": 500000.0}
  ]
}'''


def page_multi():
    """多账户路由 + 盘中分钟信号 + 定时调仓状态（第十三轮三个深化方向）。"""
    import json as _json
    import os as _os
    import sys as _sys
    import subprocess as _sp
    st.header("多账户路由 · 盘中信号 · 定时调仓")
    try:
        from lianghua.data.gateway import DataGateway
        _ok, _desc = is_trading_session()
        _ns = next_session()
        st.info("交易时段状态：**%s**；下一时段：%s" % (_desc, _ns.strftime("%Y-%m-%d %H:%M")))
    except Exception as e:
        st.error("模块加载失败：%s" % e)
        return

    _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    st.subheader("① 多账户路由（配置驱动）")
    _cfg = st.text_area("多账户配置 JSON", value=_MULTI_JSON, height=220)
    if st.button("构建并运行一次多账户调仓", key="multi_run"):
        try:
            _c = _json.loads(_cfg)
            _eng = make_multi_engine(_c)
            st.session_state["multi_status"] = _eng.step()
            st.success("已执行，见下方账户状态")
        except Exception as e:
            st.error("执行失败：%s" % e)
    if st.session_state.get("multi_status"):
        st.json(st.session_state["multi_status"])

    st.subheader("② 盘中分钟信号探针")
    _sym = st.text_input("标的代码", value="600519.SH", key="intraday_sym")
    _freq = st.selectbox("分钟频率", ["1min", "5min", "15min"], index=1)
    if st.button("探测分钟信号", key="intraday_probe"):
        try:
            _gw = DataGateway()
            _sig, _price, _df = minute_signal(_sym, "sma_cross", _gw,
                                              freq=_freq, lookback=120)
            st.write("信号：%d，最新价：%.3f，分钟线 %d 根" %
                     (_sig, _price, len(_df)))
        except Exception as e:
            st.error("探测失败：%s" % e)

    st.subheader("③ 定时调仓（automation 驱动）")
    st.caption("由 automation 周期调用 examples/run_live_cron.py，"
               "非交易时段自动跳过。")
    if st.button("立即运行一次定时调仓脚本", key="cron_run"):
        _out = _sp.run(
            [_sys.executable, "examples/run_live_cron.py",
             "--config", "live_cron.json"],
            capture_output=True, text=True, cwd=_ROOT)
        st.text_area("执行输出", _out.stdout or _out.stderr, height=200)
    _sp_path = _os.path.join(_ROOT, "live_cron_state.json")
    if _os.path.exists(_sp_path):
        try:
            _hist = _json.load(open(_sp_path, encoding="utf-8"))
            if _hist:
                st.subheader("最近一次结果")
                st.json(_hist[-1])
        except Exception:
            pass

    st.subheader("④ 微信推送测试（第十四轮）")
    _wx_key = st.text_input("企业微信群机器人 Webhook Key（可选，填 key= 后内容）",
                            value="", key="wx_key")
    if st.button("测试群机器人推送", key="wx_test") and _wx_key:
        _ok = WeChatNotifier(_wx_key).push("【量化框架】连通性测试 ✅")
        st.success("推送成功" if _ok else "推送失败（检查 key / 网络）")
    _wecom_cfg = st.text_input("企业微信自建应用配置文件路径", value="wecom_config.json",
                               key="wecom_cfg")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("测试自建应用推送", key="wecom_test"):
            try:
                _n = WeComAppNotifier(_wecom_cfg)
                _ok = _n.push("【量化框架】连通性测试 ✅")
                st.success("推送成功" if _ok else "推送失败（检查凭证/网络）")
            except Exception as e:
                st.error("加载失败：%s" % e)
    with col2:
        if st.button("执行跨账户再平衡", key="rebal_btn"):
            try:
                _c = _json.loads(_cfg)
                _eng = make_multi_engine(_c)
                _plan = _eng.rebalance(_c.get("target_weights"))
                st.session_state["rebal"] = _plan
                st.success("再平衡计划已生成")
            except Exception as e:
                st.error("再平衡失败：%s" % e)
    if st.session_state.get("rebal"):
        st.json(st.session_state["rebal"])


PAGES = {
    "单标的回测": page_single,
    "组合·篮子回测": page_basket,
    "统一编排(多资产)": page_orchestrator,
    "期权组合策略": page_option,
    "期货跨期套利": page_future_spread,
    "基金筛选": page_fund,
    "风险度量 VaR": page_var,
    "向量化回测": page_vectorized,
    "蒙特卡洛模拟": page_montecarlo,
    "参数优化": page_param,
    "HTML报告": page_html,
    "信号推送": page_notify,
    "策略库": page_strategies,
    "组合优化": page_portfolio,
    "因子研究": page_factor,
    "指标实验室": page_indicator_lab,
    "绩效风险分析": page_perf_risk,
    "数据执行能力": page_data_exec,
        "能力总览": page_capabilities,
        "实盘交易": page_live,
        "多账户与定时": page_multi,
    }
PAGES[PAGE]()
