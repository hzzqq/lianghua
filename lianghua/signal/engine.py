"""信号引擎：把策略注册到调度器，基于行情逐根 bar 评估并产出信号事件。

这是 mock 调度（纯逻辑，不依赖真实定时器），用于回测/研究阶段的信号编排。
用法：
    from lianghua.signal.engine import SignalEngine
    eng = SignalEngine().register("日内突破", IntradayBreakout())
    events = eng.run(minute_df)
"""
from __future__ import annotations

import pandas as pd

from ..strategy.intraday import IntradayStrategy


class SignalEngine:
    """注册多条策略规则，在给定行情上逐根 bar 触发，汇总信号事件。"""

    def __init__(self):
        self._rules: list[tuple[str, object, str | None]] = []

    def register(self, name: str, strategy, asset: str | None = None):
        if not (hasattr(strategy, "generate_signals") or callable(strategy)):
            raise TypeError(f"{name} 必须是策略实例或可调用")
        self._rules.append((name, strategy, asset))
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回信号事件表：[date, rule, signal, price]。"""
        events = []
        for name, strat, asset in self._rules:
            sig = strat.generate_signals(df) if hasattr(strat, "generate_signals") else strat(df)
            # 2D 输出（如单列 DataFrame）取第 0 列，避免 pd.Series(df) 报错
            if hasattr(sig, "iloc") and getattr(sig, "ndim", 1) == 2:
                sig = sig.iloc[:, 0]
            sig = pd.Series(sig)
            price = df["close"] if "close" in df else None
            for t, s in sig.items():
                try:
                    s = int(s)
                except (ValueError, TypeError):
                    # NaN / 非数值信号直接跳过，避免 int(nan) 崩溃
                    continue
                if s != 0:
                    events.append({
                        "date": str(t),
                        "rule": name,
                        "signal": s,
                        "price": float(price.loc[t]) if price is not None and t in price.index else float("nan"),
                    })
        return pd.DataFrame(events)

    def _iter_signals(self, df: pd.DataFrame):
        """逐条产出 (规则名, 时间戳, 整数信号)，跳过 0 与非数值。"""
        for name, strat, asset in self._rules:
            sig = strat.generate_signals(df) if hasattr(strat, "generate_signals") else strat(df)
            if hasattr(sig, "iloc") and getattr(sig, "ndim", 1) == 2:
                sig = sig.iloc[:, 0]
            sig = pd.Series(sig)
            for t, s in sig.items():
                try:
                    s = int(s)
                except (ValueError, TypeError):
                    continue
                if s != 0:
                    yield name, t, s

    def consensus(self, df: pd.DataFrame, min_votes: int = 1,
                  weights: dict | None = None) -> pd.Series:
        """将多条规则在 df 上的信号按日期汇总为共识信号(-1/0/1)。

        min_votes: 同向投票数达到该阈值才产生非零共识（默认 1）。
        weights:   {规则名: 权重}，默认等权；按加权求和取符号。
        这是 run() 之上新增的聚合能力，便于多策略合成单一下单信号。
        保留原始时间戳索引，便于与行情对齐。
        """
        per_date: dict = {}
        for name, t, s in self._iter_signals(df):
            w = float(weights.get(name, 1.0)) if weights else 1.0
            per_date.setdefault(t, []).append((s, w))
        out = {}
        for t, votes in per_date.items():
            wsum = sum(s * w for s, w in votes)
            n_pos = sum(1 for s, w in votes if s > 0)
            n_neg = sum(1 for s, w in votes if s < 0)
            if wsum > 0 and n_pos >= min_votes:
                out[t] = 1
            elif wsum < 0 and n_neg >= min_votes:
                out[t] = -1
            else:
                out[t] = 0
        s = pd.Series(out, name="consensus")
        # 未投票的日期补 0，输出完整时间序列，便于与行情对齐
        s = s.reindex(df.index).fillna(0).astype(int)
        return s


# 便于直接构造
__all__ = ["SignalEngine", "IntradayStrategy"]
