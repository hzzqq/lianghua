"""报告导出：回测结果 dict → CSV/JSON 文件。"""
from __future__ import annotations

import json
import math
import os

import pandas as pd


def _clean(v):
    """把单个值清洗为 JSON 安全类型：NaN/±inf → None，数值 → float，其余原样。"""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def sanitize_metrics(metrics) -> dict:
    """把任意 metrics 字典清洗为 JSON 安全的有限浮点结构。

    处理 None / NaN / ±inf / 字符串数字 / 嵌套 dict / list，避免
    ``float(v)`` 抛错或写出非法 JSON（Python 默认会把 nan 写成 "NaN"）。
    """
    if not isinstance(metrics, dict):
        return {}
    return {k: _clean(v) for k, v in metrics.items()}


def export_backtest(result: dict, path: str, name: str = "backtest"):
    """把 vectorized/orchestrator 风格的结果导出为 equity.csv + metrics.json (+ trades.csv)。"""
    os.makedirs(path, exist_ok=True)
    eq = result.get("equity")
    if isinstance(eq, pd.Series):
        eq.to_frame("equity").to_csv(os.path.join(path, f"{name}_equity.csv"), index=False)
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        with open(os.path.join(path, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(sanitize_metrics(metrics), f, ensure_ascii=False, indent=2)
    trades = result.get("trades")
    if trades is not None:
        try:
            pd.DataFrame(trades).to_csv(
                os.path.join(path, f"{name}_trades.csv"), index=False
            )
        except Exception:
            pass
    return path
