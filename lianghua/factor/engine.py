"""因子引擎：计算技术指标/量价因子，评估预测力（IC / 分层多空），并做单因子回测。

设计参考 Qlib / RD-Agent 的因子工作流：
- 用价格序列计算候选因子
- 用信息系数(IC)与分层多空收益评估因子质量
- backtest_factor 给出可投资的因子组合净值
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FactorEngine:
    """内置候选因子库，可计算 / 评估 / 回测。"""

    FACTORS = [
        "momentum_20", "reversal_5", "volatility_20", "volume_ratio_10",
        "rsi_14", "macd_hist", "boll_pctb", "bias_20", "amt_chg_5",
        "high_low_20", "close_ma5_ratio", "turnover_20",
    ]

    def __init__(self, df: pd.DataFrame):
        if "close" in df.columns:
            self.close = df["close"].astype(float)
        elif "option_price" in df.columns:
            self.close = df["option_price"].astype(float)
        else:
            self.close = df.iloc[:, 0].astype(float)
        self.volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)
        self.df = df

    # ---------- 因子计算 ----------
    def compute(self, name: str) -> pd.Series:
        c = self.close
        if name == "momentum_20":
            return c.pct_change(20)
        if name == "reversal_5":
            return -c.pct_change(5)
        if name == "volatility_20":
            return c.pct_change().rolling(20).std()
        if name == "volume_ratio_10":
            mean = self.volume.rolling(10).mean().replace(0, np.nan)
            return self.volume / mean
        if name == "rsi_14":
            return self._rsi_static(c, 14)
        if name == "macd_hist":
            return self._macd_hist_static(c)
        if name == "boll_pctb":
            return self._boll_pctb_static(c)
        if name == "bias_20":
            ma = c.rolling(20).mean().replace(0, np.nan)
            return c / ma - 1
        if name == "amt_chg_5":
            return c.pct_change(5)
        if name == "high_low_20":
            h = c.rolling(20).max()
            l = c.rolling(20).min()
            denom = (h - l).replace(0, np.nan)
            return (c - l) / denom
        if name == "close_ma5_ratio":
            ma = c.rolling(5).mean().replace(0, np.nan)
            return c / ma - 1
        if name == "turnover_20":
            return c.pct_change().abs().rolling(20).mean()
        raise ValueError(f"未知因子: {name}")

    # ---------- 因子评估 ----------
    def evaluate(self, factor: pd.Series, forward: int = 5) -> dict:
        """返回 IC、分层多空收益、样本数。"""
        if not np.isfinite(forward) or forward <= 0:
            raise ValueError("forward 必须为正整数")
        fwd = self.close.pct_change(forward).shift(-forward)
        valid = factor.notna() & fwd.notna()
        if valid.sum() < 30:
            return {"ic": float("nan"), "long_short_return": float("nan"), "n": int(valid.sum())}
        # Spearman IC = Pearson(rank(x), rank(y))，避免强依赖 scipy
        # 常数因子（零方差）相关无定义，会触发除零无效值警告 → 置 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ic_val = factor[valid].rank().corr(fwd[valid].rank())
        ic = 0.0 if pd.isna(ic_val) else float(ic_val)
        ranked = factor[valid].rank(method="first")
        # 隐性修复：因子取值过少（如常数因子）时 qcut 会抛错，用 duplicates="drop" 兜底
        try:
            groups = pd.qcut(ranked, 5, labels=False, duplicates="drop")
        except ValueError:
            return {"ic": ic, "long_short_return": float("nan"), "n": int(valid.sum())}
        if groups.nunique() < 2:
            return {"ic": ic, "long_short_return": float("nan"), "n": int(valid.sum())}
        ls = fwd[valid].groupby(groups).mean()
        long_short = float(ls.iloc[-1] - ls.iloc[0])
        return {"ic": ic, "long_short_return": long_short, "n": int(valid.sum())}

    def backtest_factor(self, factor: pd.Series, top_quantile: float = 0.2) -> pd.Series:
        """单因子回测：每期持有因子值最高的 top_quantile 比例，等权，持有至下期调仓。"""
        if not (0.0 < top_quantile <= 1.0):
            raise ValueError("top_quantile 必须介于 (0, 1]")
        pct = factor.rank(pct=True)
        signal = (pct >= 1 - top_quantile).astype(int)
        ret = self.close.pct_change().fillna(0.0)
        # 隐性修复：signal.shift(1) 首行 NaN 会让 (1+NaN*ret) 的 cumprod 全变 NaN
        shifted = signal.shift(1).fillna(0.0)
        eq = (1.0 + shifted * ret).cumprod()
        return eq

    def evaluate_all(self, forward: int = 5) -> pd.DataFrame:
        """评估全部内置因子并排序（新增能力），返回含 ic/long_short_return/n 的 DataFrame。

        便于横向比较哪些因子预测力强，是单因子逐一评估的批量可观测升级。
        """
        rows = []
        for name in self.FACTORS:
            try:
                f = self.compute(name)
            except Exception:
                continue
            res = self.evaluate(f, forward=forward)
            rows.append({
                "factor": name,
                "ic": res["ic"],
                "abs_ic": abs(res["ic"]) if np.isfinite(res["ic"]) else float("nan"),
                "long_short_return": res["long_short_return"],
                "n": res["n"],
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values("abs_ic", ascending=False, na_position="last").reset_index(drop=True)

    # ---------- 静态工具 ----------
    @staticmethod
    def _rsi_static(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50)

    @staticmethod
    def _macd_hist_static(close: pd.Series) -> pd.Series:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        return dif - dea

    @staticmethod
    def _boll_pctb_static(close: pd.Series) -> pd.Series:
        ma20 = close.rolling(20).mean()
        sd = close.rolling(20).std()
        up = ma20 + 2 * sd
        lo = ma20 - 2 * sd
        return (close - lo) / (up - lo)
