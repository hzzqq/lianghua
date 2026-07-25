"""基金多因子筛选（迭代27）。

输入一组基金的净值序列（每列一个基金，index 为日期），
计算多因子（动量 / 年化波动 / 最大回撤 / 夏普），统一方向后
加权打分，输出综合排名。不依赖网络，纯本地计算。

FundScreener.from_frame(df).screen(weights=..., method=...) -> pd.DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..perf.metrics import sharpe, max_drawdown

__all__ = ["FundScreener"]

NORMALIZERS = ("zscore", "minmax")


class FundScreener:
    """基金多因子打分筛选。"""

    FACTORS = ["momentum", "volatility", "max_drawdown", "sharpe"]
    # 各因子期望方向：1=越大越好，-1=越小越好
    DIRECTION = {"momentum": 1, "volatility": -1, "max_drawdown": -1, "sharpe": 1}

    def __init__(self, lookback: int = 60):
        if not isinstance(lookback, int) or lookback < 1:
            raise ValueError(f"lookback 必须为正整数，收到 {lookback!r}")
        self.lookback = lookback
        self._data: pd.DataFrame | None = None

    @classmethod
    def from_frame(cls, nav: pd.DataFrame, lookback: int = 60) -> "FundScreener":
        """nav: index=日期，每列为一个基金的净值序列。"""
        if not isinstance(nav, pd.DataFrame) or nav.shape[1] == 0:
            raise ValueError("nav 必须为非空的 DataFrame(每列一个基金)")
        obj = cls(lookback=lookback)
        obj._data = nav.astype(float)
        return obj

    @classmethod
    def from_series(cls, series_map: dict, lookback: int = 60) -> "FundScreener":
        if not series_map:
            raise ValueError("series_map 不能为空")
        obj = cls(lookback=lookback)
        obj._data = pd.DataFrame(series_map).astype(float)
        return obj

    def _factor_table(self) -> pd.DataFrame:
        if self._data is None or self._data.empty:
            raise ValueError("未载入基金净值数据")
        nav = self._data.dropna(how="all")
        rows = {}
        for col in nav.columns:
            s = nav[col].dropna()
            if len(s) < 2:
                rows[col] = {f: np.nan for f in self.FACTORS}
                continue
            ret = s.pct_change().dropna()
            ret = ret[np.isfinite(ret)]
            if len(ret) == 0:
                # 全为常数的净值：无法计算收益因子，给中性值
                rows[col] = {f: 0.0 for f in self.FACTORS}
                continue
            # 动量：用 lookback 窗口前的净值为基准；基准 <=0 时无法定义，置 NaN 待清洗
            base_idx = max(0, len(s) - 1 - self.lookback)
            base = s.iloc[base_idx]
            mom = float(s.iloc[-1] / base - 1.0) if base > 0 else np.nan
            vol = float(ret.std(ddof=1) * np.sqrt(252))
            mdd = max_drawdown(s)
            shp = sharpe(s)
            rows[col] = {
                "momentum": mom,
                "volatility": vol,
                "max_drawdown": mdd,
                "sharpe": shp,
            }
        return pd.DataFrame(rows, index=self.FACTORS).T

    def _sanitize(self, ft: pd.DataFrame) -> pd.DataFrame:
        """把非有限因子值(inf/nan)替换为列中位数，避免单只基金污染整体排名。"""
        out = ft.copy()
        for f in self.FACTORS:
            col = out[f]
            finite = col[np.isfinite(col)]
            fill = float(finite.median()) if len(finite) else 0.0
            out[f] = col.where(np.isfinite(col), fill)
        return out

    def screen(self, weights: dict | None = None,
               method: str = "zscore") -> pd.DataFrame:
        """返回带综合分的排名（按综合分降序）。

        method: "zscore"(默认, 对离群敏感) 或 "minmax"(对离群稳健)。
        """
        if method not in NORMALIZERS:
            raise ValueError(f"method 必须为 {NORMALIZERS}，收到 {method!r}")
        if weights is None:
            weights = {f: 1.0 for f in self.FACTORS}
        ft = self._sanitize(self._factor_table())
        norm = pd.DataFrame(index=ft.index, columns=self.FACTORS, dtype=float)
        for f in self.FACTORS:
            col = ft[f]
            if method == "minmax":
                lo, hi = col.min(), col.max()
                rng = hi - lo
                z = (col - lo) / rng if rng and rng > 0 else col * 0.0
            else:  # zscore
                mu, sd = col.mean(), col.std(ddof=0)
                z = (col - mu) / sd if sd and sd > 0 else col * 0.0
            norm[f] = z * self.DIRECTION[f]
        wsum = sum(weights.get(f, 0.0) for f in self.FACTORS)
        if wsum == 0:
            raise ValueError("权重和为 0，无法计算综合分")
        composite = sum(norm[f] * weights.get(f, 0.0) for f in self.FACTORS) / wsum
        out = ft.copy()
        out["综合分"] = composite.round(4)
        out = out.sort_values("综合分", ascending=False)
        out.insert(0, "排名", range(1, len(out) + 1))
        return out

    def top(self, n: int = 5, **kw) -> pd.DataFrame:
        """返回综合分最高的前 n 只基金。"""
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n 必须为正整数，收到 {n!r}")
        return self.screen(**kw).head(n)

    def explain(self, symbol, **kw) -> pd.Series:
        """返回某只基金的因子原始值与综合分、排名（可观测性）。"""
        df = self.screen(**kw)
        if symbol not in df.index:
            raise KeyError(f"基金 {symbol!r} 不在筛选结果中")
        row = df.loc[symbol]
        return pd.Series(
            {f: float(row[f]) for f in self.FACTORS} |
            {"综合分": float(row["综合分"]), "排名": int(row["排名"])},
            name=symbol,
        )
