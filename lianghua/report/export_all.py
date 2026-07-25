"""报告导出：回测结果 dict → CSV/JSON 文件。"""
from __future__ import annotations

import json
import os

import pandas as pd


def export_backtest(result: dict, path: str, name: str = "backtest"):
    """把 vectorized/orchestrator 风格的结果导出为 equity.csv + metrics.json。"""
    os.makedirs(path, exist_ok=True)
    eq = result.get("equity")
    if isinstance(eq, pd.Series):
        eq.to_frame("equity").to_csv(os.path.join(path, f"{name}_equity.csv"), index=False)
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        with open(os.path.join(path, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)
    return path
