"""回测/因子/蒙特卡洛结果渲染为单文件 HTML 报告。"""
from .html_report import render_backtest_html
from .tearsheet import tearsheet
from .export_all import export_backtest, sanitize_metrics

__all__ = [
    "render_backtest_html",
    "tearsheet",
    "export_backtest",
    "sanitize_metrics",
]
