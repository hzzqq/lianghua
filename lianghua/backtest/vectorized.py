"""向量化回测（迭代28）。

纯 pandas 向量化实现：信号 -> 持仓 -> 净值，无需逐行循环，
适合大样本/多标的快速扫描（作为 engine 的加速替代）。

vectorized_backtest(df, signals, init_cash=1_000_000,
                    cost_rate=0.0, shift=True) -> dict
  df: 含 close 列的 DataFrame（index 为日期）
  signals: 与 df 对齐的持仓目标 Series（1=满仓多, -1=空仓/做空, 0=空仓）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..perf.metrics import summary

__all__ = ["vectorized_backtest", "positions_from_signals", "extract_trades"]


def positions_from_signals(signals: pd.Series, shift: bool = True) -> pd.Series:
    """把信号转为实际持仓（默认次日生效，避免前视）。"""
    s = signals.reindex(signals.index).fillna(0)
    return s.shift(1).fillna(0) if shift else s


def extract_trades(positions: pd.Series) -> list[dict]:
    """从持仓序列反推成交流易（往返）列表，供绩效可观测使用（新增能力）。

    每笔交易记录：方向 side(long/short)、入场索引 entry、出场索引 exit、
    持有周期 bars。未平仓的尾部持仓不计入（避免虚高胜率）。
    """
    pos = pd.Series(positions).fillna(0.0)
    trades: list[dict] = []
    prev = 0.0
    entry_idx = None
    for i, p in pos.items():
        if p == prev:
            continue
        # 仓位发生变化：先平旧仓
        if prev != 0 and entry_idx is not None:
            trades.append({
                "side": "long" if prev > 0 else "short",
                "entry": entry_idx,
                "exit": i,
                "bars": _idx_distance(pos.index, entry_idx, i),
            })
            entry_idx = None
        # 再开新仓
        if p != 0:
            entry_idx = i
        prev = p
    return trades


def _idx_distance(index, a, b) -> int:
    try:
        return int(index.get_loc(b) - index.get_loc(a))
    except Exception:
        return 0


def vectorized_backtest(df: pd.DataFrame, signals: pd.Series,
                        init_cash: float = 1_000_000.0,
                        cost_rate: float = 0.0, shift: bool = True) -> dict:
    """向量化回测。

    返回 dict: {equity, positions, daily_returns, turnover, metrics, trades}
    - 含做空（signal=-1 时反向持仓）
    - cost_rate: 每次持仓变动(换手)按比例扣费

    强化（隐性修复）：
    - df 必须是 DataFrame 且含 close 列，否则抛清晰 ValueError（不再 KeyError）。
    - 非正价格（close<=0）置 NaN 并前向填充，避免 pct_change 产生 inf。
    - 空数据 / 全 NaN 安全返回，不崩溃。
    - cost_rate 强制 >=0，避免负成本伪装成返佣污染净值。
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df 必须是含 close 列的 DataFrame")
    if "close" not in df.columns:
        raise ValueError("df 必须包含 'close' 列")
    if not np.isfinite(init_cash) or init_cash <= 0:
        raise ValueError("init_cash 必须为正有限值")

    cost_rate = max(0.0, float(cost_rate))

    close = pd.Series(df["close"]).astype(float).reset_index(drop=True)
    # 隐性修复：非正价格会产生 inf 收益率
    close = close.mask(close <= 0, np.nan).ffill().fillna(0.0)
    if close.eq(0).all():
        # 全为零/空数据：安全返回空结果
        empty_eq = pd.Series([], dtype=float)
        return {
            "equity": empty_eq,
            "positions": pd.Series([], dtype=float).reindex(close.index),
            "daily_returns": pd.Series([], dtype=float).reindex(close.index),
            "turnover": pd.Series([], dtype=float).reindex(close.index),
            "metrics": {"final_equity": 0.0, "total_return": 0.0,
                        "annual_return": float("nan"), "sharpe": float("nan"),
                        "max_drawdown": 0.0, "calmar": float("nan"), "num_trades": 0},
            "num_changes": 0,
            "trades": [],
        }

    sig = pd.Series(signals).reset_index(drop=True).reindex(close.index).fillna(0)
    sig = sig.where(np.isfinite(sig), 0.0)
    pos = positions_from_signals(sig, shift=shift) if shift else sig

    ret = close.pct_change().fillna(0.0)
    strat_ret = pos * ret

    # 换手成本
    turnover = pos.diff().abs().fillna(pos.abs())
    if cost_rate > 0:
        strat_ret = strat_ret - turnover * cost_rate

    equity = (1 + strat_ret).cumprod() * init_cash
    if len(equity) > 0:
        equity.iloc[0] = init_cash

    # 真实成交流易（新增可观测性）
    trades = extract_trades(pos)
    num_changes = int((pos != pos.shift(1)).sum())

    metrics = summary(equity, trades=trades)
    return {
        "equity": equity,
        "positions": pos,
        "daily_returns": strat_ret,
        "turnover": turnover,
        "metrics": metrics,
        "num_changes": num_changes,
        "trades": trades,
    }
