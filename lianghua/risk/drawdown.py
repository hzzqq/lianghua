"""回撤分析：回撤序列 + 最大回撤细节 + Top-N 回撤明细表。

提供三类可观测能力：
- drawdown_series    : 与 equity 等长的逐点回撤序列（<=0，比例）。
- max_drawdown_info  : 最大回撤的起止/恢复/持续天数结构。
- drawdown_table     : Top-N 回撤区间明细（含谷底回升幅度），便于业绩复盘。

约定输入 equity 为正有限净值序列（index 为日期）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["drawdown_series", "max_drawdown_info", "drawdown_table"]


def _validate_equity(equity: pd.Series) -> pd.Series:
    """统一校验 equity，返回清洗后的正有限 Series。

    隐性修复：原 drawdown_series 在 equity 含 NaN/inf 或非正值时，
    eq/cummax 会静默产出 inf/NaN，污染下游所有回撤统计。
    """
    eq = pd.Series(equity).astype(float)
    if len(eq) == 0:
        raise ValueError("equity 为空，无法分析回撤")
    if not np.all(np.isfinite(eq)):
        raise ValueError("equity 含 NaN/inf，请先清洗数据")
    if (eq <= 0).any():
        raise ValueError("equity 必须为正（存在非正值时回撤无意义）")
    return eq


def drawdown_series(equity: pd.Series) -> pd.Series:
    """返回与 equity 等长的回撤序列（<=0，单位比例）。

    强化（隐性修复）：非有限 / 非正 equity 显式抛错；
    peak<=0（非正基准）时回撤置 0，杜绝除零 inf。
    """
    eq = _validate_equity(equity)
    peak = eq.cummax()
    dd = np.where(peak > 0, eq / peak - 1.0, 0.0)
    return pd.Series(dd, index=eq.index, name="drawdown")


def max_drawdown_info(equity: pd.Series) -> dict:
    """返回最大回撤的详细结构：深度、起止、持续天数。

    强化：
    - 空 / 单点序列安全返回 depth=0（不再 argmin 报错）。
    - 单调递增（无回撤）安全返回 depth=0、recovery 为末日。
    - 截至末日仍未收复时，duration 取实际已持续天数（原实现误置 0）。
    """
    eq = _validate_equity(equity)
    if len(eq) < 2:
        return {"max_drawdown": 0.0, "peak_date": eq.index[0],
                "trough_date": eq.index[0], "recovery_date": eq.index[0],
                "duration_days": 0}
    dd = drawdown_series(eq)
    trough_idx = int(np.argmin(dd.values))
    depth = float(dd.iloc[trough_idx])
    peak_idx = int(np.argmax(eq.iloc[: trough_idx + 1].values))
    if abs(depth) < 1e-12:
        # 无回撤：视为全程处于峰值，恢复日为末日
        return {
            "max_drawdown": 0.0, "peak_date": eq.index[0],
            "trough_date": eq.index[0], "recovery_date": eq.index[-1],
            "duration_days": 0,
        }
    rec = eq.iloc[trough_idx:][eq.iloc[trough_idx:] >= eq.iloc[peak_idx]]
    rec_idx = rec.index[0] if not rec.empty else None
    if rec_idx is not None:
        duration = list(eq.index).index(rec_idx) - peak_idx
    else:
        # 截至末日仍未收复：持续天数 = 末日 - 峰值日
        duration = len(eq) - 1 - peak_idx
    return {
        "max_drawdown": depth,
        "peak_date": eq.index[peak_idx],
        "trough_date": eq.index[trough_idx],
        "recovery_date": rec_idx,
        "duration_days": int(duration),
    }


def drawdown_table(equity: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Top-N 回撤明细表（新增可观测能力）。

    逐段识别回撤区间（创新高回落 -> 触底 -> 收复），输出每段的：
    peak_date / trough_date / recovery_date / depth / duration_days / recovery_gain。
    - depth        : 该段最大回撤（负值，比例）。
    - recovery_gain: 从谷底到收复的回升幅度（比例；未收复为 NaN）。
    - 按 |depth| 降序取前 top_n 段。
    """
    eq = _validate_equity(equity)
    if len(eq) < 2:
        raise ValueError("equity 至少需要 2 个观测值以识别回撤区间")
    if top_n <= 0:
        raise ValueError("top_n 必须为正整数")
    dd = drawdown_series(eq)
    episodes: list[dict] = []
    in_dd = False
    peak_date = None
    trough_date = None
    trough_val = 0.0
    trough_i = 0
    peak_i = 0
    for i in range(len(eq)):
        val = float(dd.iloc[i])
        if not in_dd and val < 0:
            in_dd = True
            peak_i = i - 1 if i > 0 else 0
            peak_date = eq.index[peak_i]
            trough_date = eq.index[i]
            trough_val = val
            trough_i = i
        elif in_dd:
            if val < trough_val:
                trough_val = val
                trough_date = eq.index[i]
                trough_i = i
            if val >= 0:  # 收复
                rec_gain = float(eq.iloc[i] / eq.iloc[trough_i] - 1.0)
                episodes.append({
                    "peak_date": peak_date, "trough_date": trough_date,
                    "recovery_date": eq.index[i], "depth": trough_val,
                    "duration_days": i - peak_i, "recovery_gain": rec_gain,
                })
                in_dd = False
    if in_dd:  # 末段未收复
        episodes.append({
            "peak_date": peak_date, "trough_date": trough_date,
            "recovery_date": None, "depth": trough_val,
            "duration_days": len(eq) - 1 - peak_i, "recovery_gain": float("nan"),
        })
    cols = ["peak_date", "trough_date", "recovery_date",
            "depth", "duration_days", "recovery_gain"]
    if not episodes:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(episodes).reindex(columns=cols)
    df = df.reindex(df["depth"].abs().sort_values(ascending=False).index)
    return df.head(top_n).reset_index(drop=True)
