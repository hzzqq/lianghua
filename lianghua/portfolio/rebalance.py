"""组合再平衡与回撤控制。

- rebalance：按周期把组合权重拉回目标配置；可选 drift_band 触发"漂移带"再平衡。
- drawdown_stop：组合最大回撤硬止损（回撤超阈后冻结权益，等效清仓观望）。

新增能力（v2）：
- drift_band：当前权重相对目标漂移超过阈值即触发再平衡，不再死等周期窗口。
- 输入守卫：基准价非正/非有限、未知标的、空输入等会抛清晰错误，杜绝静默 NaN/零组合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..perf.metrics import max_drawdown


def _validate_rebalance(components: pd.DataFrame, target_weights: dict):
    """再平衡输入守卫：把"静默错误"转为清晰异常。"""
    if not isinstance(components, pd.DataFrame):
        raise TypeError("components 必须是 pd.DataFrame（index=日期, columns=标的）")
    if components.empty:
        raise ValueError("components 不能为空")
    if components.shape[1] == 0:
        raise ValueError("components 至少需要一个标的列")
    if not target_weights:
        raise ValueError("target_weights 不能为空")

    base = components.iloc[0]
    bad = base[(~np.isfinite(base)) | (base <= 0)]
    if len(bad) > 0:
        cols = ", ".join(map(str, bad.index.tolist()))
        raise ValueError(
            f"标的最早一期基准价必须是有限正数，以下标的非法：{cols}"
        )

    unknown = [k for k in target_weights if k not in components.columns]
    if unknown:
        raise ValueError(f"target_weights 含未知标的（不在 components 列中）：{unknown}")


def rebalance(components: pd.DataFrame, target_weights: dict, period: int = 20,
              init_cash: float = 1_000_000.0, drift_band: float | None = None,
              return_events: bool = False):
    """按周期把组合权重拉回目标。

    components: 各子策略/标的权益矩阵（index=日期, columns=标的）。
    target_weights: {标的: 权重}。
    period: 强制再平衡周期（交易日数）。
    init_cash: 组合初始资金。
    drift_band: 漂移带阈值（0~1）。若当前权重相对目标的总变差距离超过该值，
                即便未到周期也会立即再平衡。默认 None（仅按周期再平衡）。
    return_events: True 时额外返回再平衡发生的下标列表。

    返回再平衡后的组合权益序列（金额，起点=init_cash）；return_events=True 时返回 (port, events)。
    """
    _validate_rebalance(components, target_weights)
    if drift_band is not None and not (0 < drift_band < 1):
        raise ValueError("drift_band 必须在 (0, 1) 区间")

    norm = components.div(components.iloc[0])
    if not np.all(np.isfinite(norm.values)):
        raise ValueError("components 含非有限值，归一化后出现 NaN/Inf")

    w = pd.Series(target_weights)
    w = w.reindex(components.columns).fillna(0.0)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("匹配到的标的权重之和必须为正数（检查 target_weights 取值）")
    w = w / total

    port = pd.Series(index=components.index, dtype=float)
    cur_w = w.copy()
    events: list[int] = []
    n = len(components)
    for i in range(n):
        port.iloc[i] = init_cash * float((norm.iloc[i] * cur_w).sum())
        if i == 0:
            continue
        if i % period == 0:
            cur_w = w.copy()
            events.append(i)
        elif drift_band is not None:
            s = float((norm.iloc[i] * cur_w).sum())
            if s > 0:
                cur_weights = (norm.iloc[i] * cur_w) / s
                drift = float((cur_weights - w).abs().sum() / 2.0)
                if drift > drift_band:
                    cur_w = w.copy()
                    events.append(i)

    if return_events:
        return port, events
    return port


def drawdown_stop(equity: pd.Series, max_dd: float = 0.20,
                  return_events: bool = False):
    """组合最大回撤硬止损：回撤超过 max_dd 后冻结权益（等效清仓持有现金）。"""
    eq = equity.astype(float)
    if eq.empty:
        if return_events:
            return eq, []
        return eq
    peak = eq.cummax()
    dd = eq / peak - 1.0
    stopped = dd < -abs(max_dd)
    out = eq.copy()
    frozen = None
    events: list[int] = []
    for i in range(len(out)):
        if frozen is not None:
            out.iloc[i] = frozen
            continue
        if bool(stopped.iloc[i]):
            frozen = float(out.iloc[i])
            events.append(i)
    if return_events:
        return out, events
    return out
