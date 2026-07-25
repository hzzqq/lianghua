"""交易统计：胜率 / 盈亏比 / 期望收益 / 连胜连败。

新增强化（相对迭代22基线）：
- 输入兼容 list[dict] / list[float] / np.ndarray / pd.Series。
- 新增 payoff_ratio（盈亏比）与 max_consecutive_wins / max_consecutive_losses（连胜连败）。
- 隐性修复：原 `t.get("pnl") or 0.0` 把 NaN 静默清零；现对数组/数值输入显式校验有限性，
  遇到 NaN/inf 直接抛错（可观测），仅 pnl=None（缺失）回落为 0.0 以兼容历史调用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _coerce_pnl(trades) -> np.ndarray:
    if isinstance(trades, pd.Series):
        arr = trades.to_numpy(dtype=float)
    elif isinstance(trades, np.ndarray):
        arr = trades.astype(float)
    elif isinstance(trades, (list, tuple)):
        out = []
        for t in trades:
            if isinstance(t, dict):
                v = t.get("pnl")
                if v is None:
                    v = 0.0  # 缺失成交明细回落为 0（兼容历史调用）
            else:
                v = t
            out.append(float(v))
        arr = np.array(out, dtype=float)
    else:
        raise TypeError("trades 必须是 list[dict]/list[float]/np.ndarray/pd.Series")
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("pnl 含非有限值(NaN/inf)，请先清洗数据")
    return arr


def _max_run(signs: np.ndarray):
    best_w = cur_w = 0
    best_l = cur_l = 0
    for s in signs:
        if s > 0:
            cur_w += 1
            cur_l = 0
            best_w = max(best_w, cur_w)
        elif s < 0:
            cur_l += 1
            cur_w = 0
            best_l = max(best_l, cur_l)
        else:
            cur_w = 0
            cur_l = 0
    return best_w, best_l


def trade_stats(trades) -> dict:
    """trades：成交明细（list[dict] 含 pnl 字段 / list[float] / np.ndarray / pd.Series）。

    返回汇总 dict，新增 payoff_ratio、max_consecutive_wins、max_consecutive_losses。
    """
    if len(trades) == 0:
        return {"num_trades": 0, "win_rate": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
                "payoff_ratio": 0.0, "max_consecutive_wins": 0,
                "max_consecutive_losses": 0}
    pnl = _coerce_pnl(trades)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / len(pnl) if len(pnl) else 0.0
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = -float(losses.sum()) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    payoff = (float(wins.mean()) / abs(float(losses.mean()))) if len(losses) else float("inf")
    bw, bl = _max_run(np.sign(pnl))
    return {
        "num_trades": int(len(pnl)),
        "win_rate": float(win_rate),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(pf),
        "expectancy": float(pnl.mean()),
        "payoff_ratio": float(payoff),
        "max_consecutive_wins": int(bw),
        "max_consecutive_losses": int(bl),
    }
