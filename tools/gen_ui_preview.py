"""生成 UI 主题预览（静态 HTML），复用 theme.py 的同一份 _GLOBAL_CSS，
确保预览与真实应用视觉一致。仅用于设计预览，不参与业务运行。"""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lianghua.ui import theme  # 复用同一份 CSS，避免漂移

CSS = theme._GLOBAL_CSS

HTML = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lianghua Quant · UI 主题预览</title>
<style>{CSS}
 body{{margin:0;font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;}}
 .wrap{{max-width:1100px;margin:0 auto;padding:24px 28px 40px;}}
</style></head>
<body>
<div data-testid="stAppViewContainer"><div class="wrap">

<div class="lh-hero">
  <div class="lh-hero-text">
    <div class="lh-hero-title">Lianghua Quant 多资产量化终端</div>
    <div class="lh-hero-sub">股票 · 基金 · 期货 · 期权 —— 一体化研究 / 回测 / 实盘</div>
  </div>
  <span class="lh-badge">盘中 · 连续竞价</span>
</div>

<div class="lh-kpi-grid" style="--cols:4">
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">策略</div><div class="lh-kpi-value">40</div></div>
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">优化器</div><div class="lh-kpi-value">28</div></div>
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">技术指标</div><div class="lh-kpi-value">37</div></div>
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">风险函数</div><div class="lh-kpi-value">51</div></div>
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">绩效函数</div><div class="lh-kpi-value">57</div></div>
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">因子函数</div><div class="lh-kpi-value">26</div></div>
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">期权组合</div><div class="lh-kpi-value">14</div></div>
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">数据源</div><div class="lh-kpi-value">4</div></div>
</div>

<div class="lh-sec"><div class="lh-sec-icon">🚀</div>
  <div><div class="lh-sec-title">快捷入口</div><div class="lh-sec-sub">点击直达常用模块</div></div></div>
<div class="lh-kpi-grid" style="--cols:3">
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">回测分析</div><div class="lh-kpi-value" style="font-size:16px">📈 单标的回测</div></div>
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">信号 · 执行</div><div class="lh-kpi-value" style="font-size:16px">📉 实时行情</div></div>
  <div class="lh-kpi" style="--c:#667eea"><div class="lh-kpi-label">资产 · 策略</div><div class="lh-kpi-value" style="font-size:16px">🧭 策略库</div></div>
</div>

<div class="lh-sec"><div class="lh-sec-icon">🗺️</div>
  <div><div class="lh-sec-title">能力矩阵</div><div class="lh-sec-sub">四类资产 · 一个终端</div></div></div>
<div class="lh-card"><div class="lh-card-title">覆盖资产</div>
  <div class="lh-card-body">股票 / 基金 / 期货 / 期权，统一红涨绿跌配色与数据网关。</div></div>
<div class="lh-card"><div class="lh-card-title">研究链路</div>
  <div class="lh-card-body">指标实验室 · 因子研究 · 绩效风险分析 · 组合优化，注册表驱动自动发现。</div></div>
<div class="lh-card"><div class="lh-card-title">实盘连接</div>
  <div class="lh-card-body">paper 模拟 + qmt / pt 真实柜台双保险；盘中止损 / 再平衡 / 微信推送。</div></div>

<div class="lh-tip"><span class="lh-tip-ic">💡</span>
  <div>离线环境下数据网关会自动降级为演示(假)数据，页面会明确提示；以此跑出的回测仅供功能验证，不代表真实行情。</div></div>

<div class="lh-footer"><span>Lianghua Quant · 多资产量化交易终端 ｜ 红涨绿跌 ｜ 离线降级数据为演示，仅供功能验证，不构成投资建议</span><span>v1.6.0-live</span></div>

</div></div></body></html>"""

out = ROOT / "outputs" / "ui_preview.html"
out.write_text(HTML, encoding="utf-8")
print("wrote", out)
