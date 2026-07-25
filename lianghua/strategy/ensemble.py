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
                # 隐性修复：成员信号与 df 索引完全无重叠时，reindex 会静默产出全 0
                # 信号（信号彻底丢失却不报错）。这里显式检出并抛清晰异常。
                if len(s.index.intersection(df.index)) == 0:
                    raise ValueError(
                        f"成员信号 {name!r} 索引与 df 完全无重叠，信号将全部丢失"
                        f"（请检查时间轴/频率对齐），拒绝静默集成")
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

    def reweight_by_performance(self, df: pd.DataFrame, forward_returns,
                                method: str = "accuracy",
                                alpha: float = 1.0) -> "Ensemble":
        """按各成员近期表现动态重定权重（新增能力：自适应集成）。

        原集成权重固定（用户给定），不会随市场状态调整。本方法用近期前瞻收益
        评估每个成员的方向能力，按表现重新分配权重，使集成更稳健。

        method:
        - "accuracy": 信号方向(±) 与前瞻收益方向的命中率
        - "ic":       成员分数与前瞻收益的截面相关
        - "sharpe":   信号 * 前瞻收益 模拟净值的夏普
        权重 = softmax(alpha * perf)（数值稳定版），非负且归一。
        alpha 越大越集中（赢家通吃），越小越接近等权。
        """
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha 必须为正有限值")
        if method not in ("accuracy", "ic", "sharpe"):
            raise ValueError("method 必须是 'accuracy' / 'ic' / 'sharpe'")
        fwd = pd.Series(forward_returns).astype(float).reindex(df.index).fillna(0.0)
        series = self._member_series(df)
        perfs = []
        for s, _w in series:
            sig = np.sign(s.fillna(0.0).to_numpy())
            if method == "accuracy":
                mask = sig != 0
                correct = (np.sign(fwd.to_numpy()) == sig)
                p = float(correct[mask].mean()) if mask.any() else 0.0
            elif method == "ic":
                if len(s) > 1 and float(np.std(s.to_numpy())) > 0:
                    with np.errstate(invalid="ignore"):
                        c = np.corrcoef(s.to_numpy(), fwd.to_numpy())[0, 1]
                    p = float(c) if np.isfinite(c) else 0.0
                else:
                    p = 0.0
            else:  # sharpe
                ret = sig * fwd.to_numpy()
                sd = float(np.std(ret))
                p = float(ret.mean() / sd) if sd > 0 else 0.0
            perfs.append(p)
        perfs = np.asarray(perfs, dtype=float)
        # 数值稳定 softmax
        shift = perfs.max()
        exps = np.exp(alpha * (perfs - shift))
        new_w = exps / exps.sum()
        self.members = [(name, kind, src, float(nw))
                        for (name, kind, src, _), nw in zip(self.members, new_w)]
        return self

    def performance_table(self, df: pd.DataFrame, forward_returns) -> pd.DataFrame:
        """各成员近期表现表（可观测性）：方向准确率 / IC / 模拟夏普。"""
        fwd = pd.Series(forward_returns).astype(float).reindex(df.index).fillna(0.0)
        series = self._member_series(df)
        rows = []
        for (name, _kind, _src, w), (s, _w) in zip(self.members, series):
            sig = np.sign(s.fillna(0.0).to_numpy())
            mask = sig != 0
            acc = float((np.sign(fwd.to_numpy()) == sig)[mask].mean()) if mask.any() else 0.0
            with np.errstate(invalid="ignore"):
                c = np.corrcoef(s.to_numpy(), fwd.to_numpy())[0, 1] if len(s) > 1 else np.nan
            ic = float(c) if np.isfinite(c) else 0.0
            ret = sig * fwd.to_numpy()
            sd = float(np.std(ret))
            sharpe = float(ret.mean() / sd) if sd > 0 else 0.0
            rows.append({"member": name, "weight": w, "accuracy": acc,
                         "ic": ic, "sharpe": sharpe})
        return pd.DataFrame(rows)
