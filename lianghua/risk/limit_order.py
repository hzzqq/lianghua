"""限价单回测：撮合逻辑。"""
from __future__ import annotations

import pandas as pd


def backtest_limit(orders: list, bars: pd.DataFrame) -> pd.DataFrame:
    """orders: [{side, price, date}]；bars 含 date/high/low/close。

    返回每单是否成交(filled)及成交价。
    """
    bs = bars.set_index("date") if "date" in bars else bars
    rows = []
    for o in orders:
        d = o["date"]
        if d not in bs.index:
            rows.append({**o, "filled": False, "fill_price": None})
            continue
        bar = bs.loc[d]
        px = float(o["price"])
        if o["side"] == "BUY":
            ok = float(bar["low"]) <= px
        else:
            ok = float(bar["high"]) >= px
        rows.append({**o, "filled": bool(ok), "fill_price": px if ok else None})
    return pd.DataFrame(rows)
