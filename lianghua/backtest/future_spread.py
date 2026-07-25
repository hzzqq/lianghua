"""期货跨期套利 / 价差策略（迭代26）。

同品种不同交割月份合约的价差交易：做多近月+做空远月（或反向），
基于价差 z-score 均值回归。提供：
- spread_series(near, far)           计算价差序列
- FutureSpread.run(near_df, far_df) 价差套利回测
- 信号：z>entry_z 做空价差(卖近买远)，z<-entry_z 做多价差(买近卖远)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..perf.metrics import summary

__all__ = ["spread_series", "FutureSpread"]


def spread_series(near: pd.DataFrame, far: pd.DataFrame, method: str = "near-minus-far") -> pd.Series:
    """计算价差序列（按 date 对齐）。

    method:
      near-minus-far  价差 = 近月 - 远月
      far-minus-near  价差 = 远月 - 近月
    """
    n = near.set_index("date")["close"].astype(float)
    f = far.set_index("date")["close"].astype(float)
    if method == "far-minus-near":
        return (f - n).rename("spread")
    return (n - f).rename("spread")


class FutureSpread:
    """期货跨期套利回测。"""

    def __init__(self, near_symbol: str = "", far_symbol: str = "",
                 multiplier: float = 10.0, init_cash: float = 1_000_000.0,
                 window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5,
                 lots: int = 1):
        self.near_symbol = near_symbol
        self.far_symbol = far_symbol
        self.multiplier = multiplier
        self.init_cash = init_cash
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lots = lots
        self.trades: list[dict] = []

    def signals(self, spread: pd.Series) -> pd.Series:
        """返回价差持仓方向：1=做多价差(买近卖远)，-1=做空价差(卖近买远)，0=空仓。"""
        roll = spread.rolling(self.window)
        mu = roll.mean()
        sd = roll.std().replace(0, np.nan)
        z = ((spread - mu) / sd)
        pos = pd.Series(0, index=spread.index)
        pos[z > self.entry_z] = -1   # 价差偏高 -> 做空价差
        pos[z < -self.entry_z] = 1   # 价差偏低 -> 做多价差
        # 退出：|z| 收敛到 exit_z 内
        pos[(z.abs() < self.exit_z) & (z.abs().shift(1) >= self.exit_z)] = 0
        # 仅保留状态切换（持仓持续到反向/退出）
        out = pos.where(z.abs() > self.exit_z, 0)
        # 前向填充持仓（0 表示平掉；用 ffill 维持方向直到退出）
        hold = pd.Series(0, index=spread.index, dtype=float)
        cur = 0
        for i in spread.index:
            zv = z.loc[i]
            if zv > self.entry_z:
                cur = -1
            elif zv < -self.entry_z:
                cur = 1
            elif abs(zv) < self.exit_z:
                cur = 0
            hold.loc[i] = cur
        return hold.astype(int)

    def run(self, near_df: pd.DataFrame, far_df: pd.DataFrame) -> dict:
        spread = spread_series(near_df, far_df)
        sig = self.signals(spread)
        # 仓位变化 -> 价差损益：做多价差时，spread 上行盈利
        spread_ret = spread.diff().fillna(0.0)
        pos = sig.shift(1).fillna(0).astype(int)  # t 日持仓由前一日信号决定
        # 每日盈亏 = 持仓方向 × 价差变化 × 乘数 × 手数
        daily_pnl = (pos * spread_ret * self.multiplier * self.lots)
        equity = self.init_cash + daily_pnl.cumsum()
        equity.iloc[0] = self.init_cash

        for i in spread.index:
            if sig.loc[i] != 0 and (i == spread.index[0] or sig.loc[i] != sig.loc[spread.index[spread.index.get_loc(i)-1]]):
                self.trades.append({
                    "date": str(i), "side": "LONG_SPREAD" if sig.loc[i] == 1 else "SHORT_SPREAD",
                    "spread": round(float(spread.loc[i]), 3),
                    "z": None,
                })

        metrics = summary(equity, trades=self.trades)
        return {
            "spread": spread,
            "signals": sig,
            "equity": equity,
            "daily_pnl": daily_pnl,
            "metrics": metrics,
            "trades": self.trades,
        }
