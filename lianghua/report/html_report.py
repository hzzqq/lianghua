"""单文件 HTML 回测报告（内联 CSS + SVG 权益曲线 + 指标卡 + 交易表）。

不依赖 Plotly：权益曲线用纯 SVG 折线，保证单文件可直接打开。
"""
from __future__ import annotations

import html

from ..perf.metrics import report as perf_report
from ..backtest.montecarlo import bootstrap


def _finite(v) -> bool:
    """判断值是否为有限实数（排除 None / NaN / ±inf，以及无法转为浮点的类型）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and abs(f) != float("inf")


def _fmt_pct(v, nd: int = 1) -> str:
    return f"{v * 100:.{nd}f}%" if _finite(v) else "N/A"


def _fmt_num(v, nd: int = 2) -> str:
    return f"{v:.{nd}f}" if _finite(v) else "N/A"


def _equity_svg(equity: "object", w: int = 800, h: int = 240) -> str:
    vals = [float(x) for x in equity.values.astype(float)]
    finite = [v for v in vals if _finite(v)]
    if len(finite) < 2:
        return (
            f'<svg viewBox="0 0 {w} {h}" width="100%" '
            f'style="background:#12122a;border-radius:8px">'
            f'<text x="20" y="{h // 2}" fill="#9aa0c0">无有效权益数据</text></svg>'
        )
    lo, hi = min(finite), max(finite)
    if hi == lo:
        hi = lo + 1.0
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        # 非有限点落到基线，避免 polyline 出现 NaN 坐标而整条线断裂
        fv = v if _finite(v) else lo
        x = 10 + (w - 20) * (i / max(1, n - 1))
        y = h - 10 - (h - 20) * ((fv - lo) / (hi - lo))
        pts.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="none" '
        f'style="background:#12122a;border-radius:8px">'
        f'<polyline fill="none" stroke="#667eea" stroke-width="2" points="{" ".join(pts)}"/>'
        f"</svg>"
    )


def render_backtest_html(
    result,
    symbol,
    strategy,
    asset=None,
    init_cash: float = 1_000_000.0,
    with_mc: bool = False,
) -> str:
    """渲染回测结果为完整 HTML 文档字符串。"""
    trades = result.trades or []
    try:
        p = perf_report(result.equity, trades, init_cash=init_cash)
        if not isinstance(p, dict):
            p = {}
    except Exception:
        # equity 含 NaN/非正值等导致 perf_report 失败时，降级为全 N/A 卡片
        p = {}
    mc_html = ""
    if with_mc:
        try:
            mc = bootstrap(result.equity, n=300, seed=1)
            mc_html = (
                "<h3>蒙特卡洛稳健性（%d 次重采样）</h3><ul>"
                "<li>中位收益：%.1f%%</li>"
                "<li>P5 ~ P95：%.1f%% ~ %.1f%%</li>"
                "<li>亏损概率：%.1f%%</li></ul>"
                % (
                    mc["n_simulations"],
                    mc["median_return"] * 100,
                    mc["p5"] * 100,
                    mc["p95"] * 100,
                    mc["prob_loss"] * 100,
                )
            )
        except Exception:
            mc_html = ""
    rows = "".join(
        "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
        % (
            html.escape(str(t.get("date", ""))),
            html.escape(str(t.get("side", ""))),
            html.escape(str(t.get("price", ""))),
            html.escape(str(t.get("qty", ""))),
        )
        for t in (result.trades or [])[:100]
    )
    cards = "".join(
        '<div class="card"><div class="k">%s</div><div class="v">%s</div></div>' % (k, v)
        for k, v in [
            ("总收益", _fmt_pct(p.get("total_return", float("nan")))),
            ("年化", _fmt_pct(p.get("annual_return", float("nan")))),
            ("夏普", _fmt_num(p.get("sharpe", float("nan")))),
            ("最大回撤", _fmt_pct(p.get("max_drawdown", float("nan")))),
            ("交易次数", "%d" % int(p.get("num_trades", 0) or 0)),
        ]
    )
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Lianghua 回测报告 {html.escape(str(symbol))} {html.escape(str(strategy))}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,'Microsoft YaHei',sans-serif;background:#0f0f23;color:#e6e6ef;margin:0;padding:24px}}
h1{{color:#fff}} .cards{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}
.card{{background:#1a1a2e;border-radius:12px;padding:14px 18px;min-width:120px}}
.card .k{{color:#9aa0c0;font-size:13px}} .card .v{{font-size:22px;font-weight:700;color:#667eea}}
table{{width:100%;border-collapse:collapse;margin-top:12px}} th,td{{border-bottom:1px solid #2a2a44;padding:6px 10px;font-size:13px;text-align:left}}
th{{color:#9aa0c0}}
</style></head><body>
<h1>Lianghua 回测报告</h1>
<p>标的 <b>{html.escape(str(symbol))}</b> · 策略 <b>{html.escape(str(strategy))}</b> · 资产 <b>{html.escape(str(asset or 'auto'))}</b></p>
<div class="cards">{cards}</div>
<h3>权益曲线</h3>{_equity_svg(result.equity)}
{mc_html}
<h3>交易明细（前 {min(100, len(result.trades or []))} 笔）</h3>
<table><tr><th>日期</th><th>方向</th><th>价格</th><th>数量</th></tr>{rows}</table>
</body></html>"""
