"""Lianghua Quant —— 前端主题与可复用组件层。

纯展示层，不改动任何业务/回测逻辑。目标：
- 在原生暗色主题（.streamlit/config.toml）之上，注入统一抛光 CSS；
- 提供自包含配色的组件（KPI 瓦片 / 区块标题 / 卡片 / 提示 / 首页英雄区 / 页脚），
  保证在任何基底下都可读、风格一致、有记忆点。
"""
from __future__ import annotations

import html
from typing import Iterable, Mapping, Sequence

import streamlit as st

# —— 项目调色板（与 starfield 主题一致）——
BG = "#0f0f23"
CARD = "#1a1a2e"
BORDER = "rgba(102,126,234,.22)"
ACCENT = "#667eea"
ACCENT2 = "#764ba2"
UP = "#ff4d4f"      # 涨 · 红
DOWN = "#00d486"    # 跌 · 绿
TEXT = "#e6e8f0"
MUTED = "#9aa0c0"


def _esc(v) -> str:
    """HTML 转义，避免动态文本破坏结构。"""
    return html.escape(str(v))


# ---------------------------------------------------------------------------
# 全局抛光 CSS（注入一次）
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
:root{
  --lh-bg:#0f0f23; --lh-card:#1a1a2e; --lh-border:rgba(102,126,234,.22);
  --lh-accent:#667eea; --lh-accent2:#764ba2;
  --lh-up:#ff4d4f; --lh-down:#00d486; --lh-text:#e6e8f0; --lh-muted:#9aa0c0;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 88% -12%, rgba(118,75,162,.20), transparent 60%),
    radial-gradient(900px 520px at -6% 8%, rgba(102,126,234,.18), transparent 55%),
    var(--lh-bg);
  color:var(--lh-text);
}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(26,26,46,.96), rgba(15,15,35,.96));
  border-right:1px solid var(--lh-border);
}
[data-testid="stHeader"]{ background:transparent; border-bottom:1px solid var(--lh-border); }
h1{ font-weight:800 !important; letter-spacing:.4px; }
h2{ color:var(--lh-text); border-left:3px solid var(--lh-accent); padding-left:10px; margin-top:18px; }
h3{ color:#c9cdf0; }

/* Metric 卡片化 */
[data-testid="stMetric"]{
  background:linear-gradient(180deg, rgba(26,26,46,.6), rgba(26,26,46,.35));
  border:1px solid var(--lh-border); border-radius:14px; padding:10px 14px;
}
[data-testid="stMetricLabel"]{ color:var(--lh-muted); font-size:12px;
  text-transform:uppercase; letter-spacing:.5px; }
[data-testid="stMetricValue"]{ color:#fff; }
[data-testid="stMetricDelta"]{ color:var(--lh-muted); }

/* 按钮强调 */
[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,var(--lh-accent),var(--lh-accent2)) !important;
  border:none !important; border-radius:10px !important; font-weight:700;
}
[data-testid="stBaseButton-secondary"]{ border-radius:10px !important; }

/* 滚动条 */
*::-webkit-scrollbar{ width:10px; height:10px; }
*::-webkit-scrollbar-track{ background:rgba(255,255,255,.03); }
*::-webkit-scrollbar-thumb{ background:rgba(102,126,234,.4); border-radius:6px; }
*::-webkit-scrollbar-thumb:hover{ background:rgba(102,126,234,.7); }

/* DataFrame / 表格圆角 */
.stDataFrame, [data-testid="stTable"]{ border-radius:12px; }

/* —— 组件类 —— */
.lh-kpi-grid{ display:grid; grid-template-columns:repeat(var(--cols,4),minmax(0,1fr));
  gap:14px; margin:10px 0 4px; }
.lh-kpi{ position:relative; overflow:hidden;
  background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.015));
  border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:14px 16px; }
.lh-kpi::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--c,var(--lh-accent)); }
.lh-kpi-label{ font-size:12px; color:var(--lh-muted); letter-spacing:.4px; }
.lh-kpi-value{ font-size:23px; font-weight:800; color:#fff; margin-top:4px; line-height:1.1; }
.lh-kpi-sub{ font-size:12px; color:var(--c,var(--lh-accent)); margin-top:4px; font-weight:600; }

.lh-sec{ display:flex; align-items:center; gap:12px; margin:18px 0 10px; }
.lh-sec-icon{ width:40px; height:40px; flex:0 0 40px; border-radius:12px;
  display:flex; align-items:center; justify-content:center; font-size:20px;
  background:linear-gradient(135deg,var(--lh-accent),var(--lh-accent2));
  box-shadow:0 4px 14px rgba(102,126,234,.35); }
.lh-sec-title{ font-size:18px; font-weight:800; color:#fff; }
.lh-sec-sub{ font-size:13px; color:var(--lh-muted); margin-top:2px; }

.lh-card{ background:linear-gradient(180deg, rgba(26,26,46,.7), rgba(26,26,46,.45));
  border:1px solid var(--lh-border); border-radius:16px; padding:16px 18px; margin:12px 0; }
.lh-card-title{ font-size:15px; font-weight:700; color:#fff; margin-bottom:8px; }
.lh-card-body{ color:var(--lh-text); font-size:14px; }

.lh-tip{ display:flex; gap:10px; align-items:flex-start;
  background:rgba(102,126,234,.1); border:1px solid var(--lh-border);
  border-left:4px solid var(--lh-accent); border-radius:12px;
  padding:10px 14px; color:var(--lh-text); font-size:13px; margin:10px 0; }
.lh-tip-ic{ font-size:16px; }

.lh-hero{ display:flex; justify-content:space-between; align-items:center; gap:16px;
  background:linear-gradient(135deg, rgba(102,126,234,.18), rgba(118,75,162,.14));
  border:1px solid var(--lh-border); border-radius:18px; padding:22px 24px; margin:6px 0 14px; }
.lh-hero-title{ font-size:30px; font-weight:900;
  background:linear-gradient(135deg,#9db4ff,#c4a6ff); -webkit-background-clip:text;
  background-clip:text; color:transparent; }
.lh-hero-sub{ color:var(--lh-muted); margin-top:6px; font-size:14px; }
.lh-badge{ background:rgba(0,212,134,.15); border:1px solid rgba(0,212,134,.4);
  color:#00d486; padding:6px 12px; border-radius:999px; font-size:13px; font-weight:700; white-space:nowrap; }

.lh-footer{ margin-top:26px; padding-top:14px; border-top:1px solid var(--lh-border);
  color:var(--lh-muted); font-size:12px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }

/* 数据来源徽标 */
.lh-src{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; }
.lh-src-real{ background:rgba(0,212,134,.15); border:1px solid rgba(0,212,134,.4); color:#00d486; }
.lh-src-demo{ background:rgba(255,77,79,.15); border:1px solid rgba(255,77,79,.4); color:#ff4d4f; }
.lh-src-unknown{ background:rgba(154,160,192,.15); border:1px solid rgba(154,160,192,.4); color:#9aa0c0; }

/* 空状态 */
.lh-empty{ text-align:center; color:var(--lh-muted); padding:24px;
  border:1px dashed var(--lh-border); border-radius:14px; margin:10px 0; }

/* 图表面板（把 plotly 包进卡片，统一观感） */
.lh-panel{ background:linear-gradient(180deg, rgba(26,26,46,.55), rgba(26,26,46,.32));
  border:1px solid var(--lh-border); border-radius:16px; padding:10px 12px; margin:10px 0; }

/* 状态徽标（ok/warn/info 三色胶囊） */
.lh-badge-ok{ background:rgba(0,212,134,.15); border:1px solid rgba(0,212,134,.4);
  color:#00d486; padding:5px 12px; border-radius:999px; font-size:13px; font-weight:700; white-space:nowrap; }
.lh-badge-warn{ background:rgba(255,176,32,.15); border:1px solid rgba(255,176,32,.45);
  color:#ffb020; padding:5px 12px; border-radius:999px; font-size:13px; font-weight:700; white-space:nowrap; }
.lh-badge-info{ background:rgba(102,126,234,.15); border:1px solid rgba(102,126,234,.4);
  color:#9db4ff; padding:5px 12px; border-radius:999px; font-size:13px; font-weight:700; white-space:nowrap; }
.lh-badge-danger{ background:rgba(255,77,79,.15); border:1px solid rgba(255,77,79,.45);
  color:#ff4d4f; padding:5px 12px; border-radius:999px; font-size:13px; font-weight:700; white-space:nowrap; }

/* 数据源 chip（小标签） */
.lh-chip{ display:inline-block; margin:2px 4px 2px 0; padding:3px 9px; border-radius:8px;
  background:rgba(102,126,234,.12); border:1px solid var(--lh-border); color:#c9cdf0; font-size:12px; }

/* 回到顶部 / 回首页 按钮区 */
.lh-navbtn{ margin:6px 0; }
"""


def inject_theme() -> None:
    """注入全局抛光 CSS（每个 run 调用一次即可）。"""
    st.markdown(f"<style>{_GLOBAL_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 组件
# ---------------------------------------------------------------------------
def kpi_grid(items: Sequence[Mapping], columns: int = 4) -> None:
    """渲染一排自包含配色的 KPI 瓦片。

    items: [{"label","value","sub"?, "color"?}, ...]
      color 缺省为强调色；sub 用于副文本（如方向提示）。
    """
    cells = []
    for it in items:
        color = it.get("color") or "var(--lh-accent)"
        sub = it.get("sub")
        sub_html = f'<div class="lh-kpi-sub">{_esc(sub)}</div>' if sub not in (None, "") else ""
        cells.append(
            f'<div class="lh-kpi" style="--c:{color}">'
            f'<div class="lh-kpi-label">{_esc(it.get("label", ""))}</div>'
            f'<div class="lh-kpi-value">{_esc(it.get("value", ""))}</div>'
            f'{sub_html}</div>'
        )
    html_block = (
        f'<div class="lh-kpi-grid" style="--cols:{int(columns)}">'
        + "".join(cells)
        + "</div>"
    )
    st.markdown(html_block, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", icon: str = "📊") -> None:
    sub_html = f'<div class="lh-sec-sub">{_esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="lh-sec"><div class="lh-sec-icon">{_esc(icon)}</div>'
        f'<div><div class="lh-sec-title">{_esc(title)}</div>{sub_html}</div></div>',
        unsafe_allow_html=True,
    )


def card(title: str = "", body: str = "", accent: str | None = None) -> None:
    ac = accent or "var(--lh-accent)"
    title_html = f'<div class="lh-card-title">{_esc(title)}</div>' if title else ""
    st.markdown(
        f'<div class="lh-card" style="--c:{ac}">{title_html}'
        f'<div class="lh-card-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def tip(text: str, icon: str = "💡") -> None:
    st.markdown(
        f'<div class="lh-tip"><span class="lh-tip-ic">{_esc(icon)}</span>'
        f'<div>{_esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str = "", badge: str | None = None) -> None:
    badge_html = f'<span class="lh-badge">{_esc(badge)}</span>' if badge else ""
    sub_html = f'<div class="lh-hero-sub">{_esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="lh-hero"><div class="lh-hero-text">'
        f'<div class="lh-hero-title">{_esc(title)}</div>{sub_html}</div>'
        f'{badge_html}</div>',
        unsafe_allow_html=True,
    )


def render_footer(version: str = "1.6.0-live") -> None:
    st.markdown(
        f'<div class="lh-footer"><span>Lianghua Quant · 多资产量化交易终端'
        f'　|　红涨绿跌　|　离线降级数据为演示，仅供功能验证，不构成投资建议</span>'
        f'<span>v{_esc(version)}</span></div>',
        unsafe_allow_html=True,
    )


def source_badge(source: str, was_demo: bool = False) -> None:
    """渲染数据来源徽标：真实(绿) / 演示(红) / 未知(灰)。

    与项目「降级可观测、绝不把假数据当真」铁律一致，让用户一眼看清当前价/图是否可信。
    """
    src = (source or "").lower()
    if was_demo or src == "demo":
        cls, txt = "demo", "演示数据"
    elif src in ("akshare", "cache", "baostock", "last_close"):
        cls, txt = "real", f"真实 · {src}"
    else:
        cls, txt = "unknown", (source or "未知")
    st.markdown(f'<span class="lh-src lh-src-{cls}">{_esc(txt)}</span>', unsafe_allow_html=True)


def empty_state(text: str, icon: str = "🗂️") -> None:
    """渲染统一的空状态提示（虚线框 + 居中），避免页面出现「空白尴尬」。"""
    st.markdown(f'<div class="lh-empty">{_esc(icon)} &nbsp; {_esc(text)}</div>',
                unsafe_allow_html=True)


def chart_panel(title: str, fig, height: int | None = None, use_container_width: bool = True) -> None:
    """把一张 Plotly 图包进统一样式卡片里渲染。

    比裸 ``st.plotly_chart`` 多一层卡片边框，让各页图表观感一致；
    title 仅作语义说明（图本身已自带 title 时可用空串）。
    """
    if title:
        section_header(title, icon="📈")
    st.markdown('<div class="lh-panel">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=use_container_width,
                    height=height, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "ok") -> None:
    """渲染状态徽标胶囊。

    kind: "ok"(绿) / "warn"(橙) / "info"(蓝) / "danger"(红)。
    用于后端连通、交易时段、运行结果等状态一眼可辨。
    """
    cls = {
        "ok": "lh-badge-ok", "warn": "lh-badge-warn",
        "info": "lh-badge-info", "danger": "lh-badge-danger",
    }.get(kind, "lh-badge-info")
    st.markdown(f'<span class="{cls}">{_esc(text)}</span>', unsafe_allow_html=True)


def chip(text: str) -> None:
    """渲染一个数据源/标签小芯片。"""
    st.markdown(f'<span class="lh-chip">{_esc(text)}</span>', unsafe_allow_html=True)


def csv_export(df, filename: str, label: str = "⬇ 导出 CSV") -> None:
    """为任意 DataFrame 提供一键 CSV 下载按钮（内存生成，不落盘）。"""
    try:
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    except Exception:
        csv_bytes = df.to_csv(index=True).encode("utf-8-sig")
    st.download_button(label=label, data=csv_bytes,
                        file_name=filename, mime="text/csv")


def back_to_home() -> None:
    """在侧边栏/页面里放一个「回首页」按钮（on_click 跳转）。"""
    st.button("🏠 回首页", key="lh_back_home", on_click=_goto_home,
              help="返回仪表盘首页")


def _goto_home():
    import streamlit as _st
    _st.session_state["lh_cat"] = "仪表盘"
    _st.session_state["lh_page"] = "首页"


def auto_refresh(seconds: int = 10) -> None:
    """可选自动刷新：在「自动刷新」开关打开时，阻塞若干秒后 rerun。

    仅在用户显式开启时调用（放在按钮分支里），避免无脑轮询；
    rerun 缺失（如测试桩）时静默跳过，不会死循环。
    """
    import time as _t
    try:
        _t.sleep(int(seconds))
        if hasattr(st, "rerun"):
            st.rerun()
    except Exception:
        pass


def sparkline(values, color: str = "#667eea", width: int = 120, height: int = 36,
              label: str = "", up: bool | None = None) -> None:
    """渲染迷你走势线（内联 SVG，零依赖）。用于首页快捷卡片等「动态感」场景。

    up=None 时按序列首尾自动判断红/绿（红涨绿跌）；也可显式指定 up=True/False。
    数据为空时优雅降级为占位符，绝不渲染破图。
    """
    try:
        xs = [float(v) for v in values]
    except Exception:
        xs = []
    if not xs or len(xs) < 2:
        st.markdown('<span style="color:var(--lh-muted)">—</span>', unsafe_allow_html=True)
        return
    _col = color
    if up is None:
        up = xs[-1] >= xs[0]
    if up is not None:
        _col = UP if up else DOWN
    n = len(xs)
    _min, _max = min(xs), max(xs)
    _span = (_max - _min) or 1.0
    _pad = 3
    _w = width - 2 * _pad
    _h = height - 2 * _pad
    pts = []
    for i, v in enumerate(xs):
        _x = _pad + (_w * i / (n - 1)) if n > 1 else _pad + _w / 2
        _y = _pad + _h * (1 - (v - _min) / _span)
        pts.append((_x, _y))
    _d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    _area = (f"{_d} L {pts[-1][0]:.1f},{height - _pad} "
             f"L {pts[0][0]:.1f},{height - _pad} Z")
    _last = pts[-1]
    _svg = (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="sparkline">'
        f'<path d="{_area}" fill="{_col}" fill-opacity="0.15"/>'
        f'<path d="{_d}" fill="none" stroke="{_col}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{_last[0]:.1f}" cy="{_last[1]:.1f}" r="2.2" fill="{_col}"/>'
        f'</svg>'
    )
    _label_html = (f'<div style="font-size:11px;color:var(--lh-muted);'
                   f'margin-top:2px">{_esc(label)}</div>') if label else ""
    st.markdown(f'<div style="display:inline-block">{_svg}{_label_html}</div>',
                unsafe_allow_html=True)

