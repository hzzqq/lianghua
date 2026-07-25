"""信号调度：按交易日历周期触发信号评估。"""
from __future__ import annotations

import pandas as pd


def rebalance_dates(start, end, period: str = "M") -> pd.DatetimeIndex:
    """返回 start~end 间每个 period（M=月/W=周/Q=季）末的再平衡日。"""
    s = pd.date_range(start, end, freq="B")
    return s[s.to_series().dt.to_period(period).astype(str).duplicated(keep="last")]


def schedule(signal_fn, universe: list, calendar: pd.DatetimeIndex,
             step: int = 5) -> pd.DataFrame:
    """在 calendar 中按 step 间隔取评估日，对 universe 各标的调用 signal_fn(sym, date)。

    返回 index=评估日, columns=标的, 值=signal。
    """
    cal = pd.to_datetime(calendar)
    chosen = cal[::max(1, step)]
    out = {sym: [signal_fn(sym, d) for d in chosen] for sym in universe}
    return pd.DataFrame(out, index=chosen)
