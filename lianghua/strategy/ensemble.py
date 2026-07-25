"""多策略集成：把多个策略信号按权重合成为组合信号。

用法：
    from lianghua.strategy.ensemble import Ensemble
    ens = Ensemble().add("均线", "sma_cross", 1.0).add("动量", "momentum", 0.5)
    sig = ens.signals(df, threshold=0.3)

增强（本轮）：
- 支持三种合成方式：weighted（加权均值）/ vote（加权投票）/ zscore（z 分数合成）。
- 支持直接加入预计算信号 add_signal（不限于策略对象）。
- 隐性修复：成员信号索引与 df 不一致时原实现会按索引对齐产生错位/NaN；
  非有限权重、NaN 信号、NaN 阈值均显式拒绝。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .examples import get_strategy

__all__ = ["Ensemble"]


class Ensemble:
    """加权集成多个策略（策略名、策略实例或预计算信号均可）。"""

    def __init__(self):
        self.members: list[tuple[str, str, object, float]] = []
        # (name, kind∈{"strategy","signal"}, source, weight)

    @staticmethod
    def _check_weight(weight: float) -> float:
        w = float(weight)
        if not np.isfinite(w):
            raise ValueError("weight 必须为有限数")
        return w

    def add(self, name: str, strategy, weight: float = 1.0):
        if isinstance(strategy, str):
            strategy = get_strategy(strategy)
        if not hasattr(strategy, "generate_signals"):
            raise TypeError(f"{name} 不是合法策略（缺少 generate_signals）")
        self.members.append((name, "strategy", strategy, self._check_weight(weight)))
        return self

    def add_signal(self, name: str, signal, weight: float = 1.0):
        sig = pd.Series(signal).astype(float)
        if not np.isfinite(sig.to_numpy()).all():
            raise ValueError(f"{name} 信号含 NaN/inf，无法集成")
        self.members.append((name, "signal", sig, self._check_weight(weight)))
        return self

    def weights(self) -> dict:
        """返回各成员归一化权重（可观测）。"""
        total = sum(w for _, _, _, w in self.members) or 1.0
        return {name: w / total for name, _, _, w in self.members}

    def _member_series(self, df: pd.DataFrame) -> list[tuple[pd.Series, float]]:
        out = []
        for name, kind, src, w in self.members:
            if kind == "strategy":
                s = src.generate_signals(df).astype(float)
            else:
                s = src.astype(float)
            # 隐性修复：按 df 索引重对齐，缺失补 0，杜绝错位/NaN
            s = s.reindex(df.index, fill_value=0.0)
            out.append((s, w))
        return out

    def raw(self, df: pd.DataFrame, method: str = "weighted") -> pd.Series:
        """返回合成前的原始组合分数（可观测），不做阈值离散化。"""
        if not self.members:
            raise ValueError("集成器为空，请先 add()/add_signal() 添加成员")
        if method not in ("weighted", "vote", "zscore"):
            raise ValueError("method 必须是 'weighted' / 'vote' / 'zscore'")
        series = self._member_series(df)
        total = sum(w for _, w in series) or 1.0
        if method == "weighted":
            combo = sum(s * (w / total) for s, w in series)
        elif method == "vote":
            combo = sum(np.sign(s.fillna(0.0)) * (w / total) for s, w in series)
        else:  # zscore
            zs = []
            for s, w in series:
                sd = float(s.std(ddof=0))
                z = (s - float(s.mean())) / sd if sd > 0 else s * 0.0
                zs.append(z * (w / total))
            combo = sum(zs)
        return pd.Series(combo, index=df.index, name="ensemble_raw")

    def signals(self, df: pd.DataFrame, threshold: float = 0.3,
                method: str = "weighted") -> pd.Series:
        if not np.isfinite(threshold):
            raise ValueError("threshold 必须为有限数")
        combo = self.raw(df, method=method)
        sig = pd.Series(0, index=df.index)
        sig[combo > threshold] = 1
        sig[combo < -threshold] = -1
        return sig
