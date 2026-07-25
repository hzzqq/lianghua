"""统一策略注册表：把「类式策略(examples)」与「函数式策略(迭代31-100)」合并为单一入口。

- 旧策略（sma_cross / macd / momentum / breakout / mean_reversion）来自 strategy/examples，
  以 StrategyBase 子类形式注册。
- 新策略（turtle / rsi / bollinger / grid / zscore / donchian / keltner / coppock /
  vol_breakout / rsi_divergence / sar / ou / ml / adaptive / seasonal）来自迭代31-100，
  是函数式信号生成器，签名各异（有的吃 df、有的吃 close、有的吃 high/low）。
  这里用 FnStrategy 统一包成 StrategyBase，并按 kind 适配入参。

对外只暴露：
    from lianghua.strategy.registry import get_strategy, STRATEGY_NAMES, STRATEGY_INFO
runner / orchestrator / UI 全部走这个入口，策略数量随迭代自动增长。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyBase
from .examples import REGISTRY as EXAMPLES_REGISTRY, get_strategy as _get_examples

# ---- 函数式策略适配 ----------------------------------------------------------
# kind 决定 generate_signals 如何把 df 喂给底层函数：
#   df      -> fn(df, **kw)
#   close   -> fn(df["close"], **kw)
#   hl      -> fn(df["high"], df["low"], **kw)
#   hlc     -> fn(df["high"], df["low"], df["close"], **kw)
#   close_rsi -> fn(df["close"], rsi(df["close"]), **kw)
#   ret     -> fn(df["close"].pct_change().fillna(0), **kw)
#   grid    -> fn(df, **kw) 再把档位符号化(-1/0/1)
_FN_SPECS = {
    "turtle":        ("df", "海龟通道突破（唐奇安+ATR）"),
    "rsi":           ("df", "RSI 超买超卖（30/70）"),
    "bollinger":     ("df", "布林带均值回复"),
    "grid":          ("grid", "网格交易（档位符号化为方向）"),
    "zscore":        ("df", "Z-score 均值回归"),
    "donchian":      ("close", "唐奇安通道突破"),
    "keltner":       ("hlc", "Keltner 通道突破"),
    "coppock":       ("close", "考普克长短周期动量"),
    "vol_breakout":  ("close", "波动率突破"),
    "rsi_divergence":("close_rsi", "RSI 顶/底背离"),
    "sar":           ("hl", "抛物线 SAR 趋势"),
    "ou":            ("close", "OU 均值回复"),
    "ml":            ("df", "机器学习择时（LR，失败降级动量）"),
    "adaptive":      ("ret", "自适应波动仓位"),
    "seasonal":      ("close", "季节性月份效应"),
    # 迭代 111-120：进阶技术策略
    "supertrend":    ("df", "Supertrend 趋势跟踪（ATR）"),
    "aroon":         ("df", "Aroon 趋势强度"),
    "vortex":        ("df", "Vortex 指标交叉"),
    "trix":          ("close", "TRIX 三重平滑动量交叉"),
    "williams_r":    ("df", "Williams %R 超买超卖反转"),
    "elder_ray":     ("df", "Elder Ray 多空力量"),
    "ichimoku":      ("df", "一目均衡表云图"),
    "macd_hist":     ("close", "MACD 柱状图零轴穿越"),
    "chaikin":       ("df", "Chaikin 资金流(CMF)"),
    "stoch_rsi":     ("close", "Stochastic RSI 超买超卖"),
    # 迭代 131-135：进阶技术策略
    "parabolic_sar": ("df", "抛物线 SAR 趋势"),
    "adx_trend":     ("df", "ADX 趋势强度(顺势)"),
    "cci_signal":    ("df", "CCI 均值回复"),
    "roc":           ("close", "ROC 变动率动量"),
    "ultimate_oscillator": ("df", "终极波动指标(超买超卖)"),
    # 迭代 141-145：进阶技术策略
    "pivot_points":  ("df", "轴心点(Pivot)多空分界"),
    "heikin_ashi":   ("df", "Heikin-Ashi 云图趋势"),
    "renko_trend":   ("df", "Renko 砖形趋势跟踪"),
    "demark":        ("df", "DeMark 比较动量"),
    "klinger":       ("df", "Klinger 量价振荡器"),
}


class FnStrategy(StrategyBase):
    """把任意函数式信号生成器包成 StrategyBase。"""

    def __init__(self, fn, kind: str = "df", description: str = "", **kw):
        self._fn = fn
        self._kind = kind
        self._desc = description
        self._kw = kw

    @property
    def name(self) -> str:
        return getattr(self._fn, "__name__", "fn_strategy")

    @property
    def description(self) -> str:
        return self._desc

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        kind = self._kind
        if kind == "df":
            sig = self._fn(df, **self._kw)
        elif kind == "close":
            sig = self._fn(df["close"], **self._kw)
        elif kind == "hl":
            sig = self._fn(df["high"], df["low"], **self._kw)
        elif kind == "hlc":
            sig = self._fn(df["high"], df["low"], df["close"], **self._kw)
        elif kind == "close_rsi":
            from ..indicators.tech import rsi
            sig = self._fn(df["close"], rsi(df["close"]), **self._kw)
        elif kind == "ret":
            sig = self._fn(df["close"].pct_change().fillna(0), **self._kw)
        elif kind == "grid":
            levels = self._fn(df, **self._kw)
            sig = np.sign(pd.Series(levels)).fillna(0)
        else:
            sig = self._fn(df, **self._kw)
        return pd.Series(sig).reindex(df.index).fillna(0)


# ---- 构建统一注册表 ----------------------------------------------------------
def _build_fn_registry():
    reg = {}
    info = {}
    # 延迟导入，避免循环依赖
    from . import (
        turtle, rsi_strategy, bollinger, grid, mean_reversion_z, donchian,
        keltner, coppock, vol_breakout, rsi_divergence, parallel_sar,
        ou_mean_reversion, ml_signal, adaptive, seasonal,
        supertrend, aroon, vortex, trix_strategy, williams_r,
        elder_ray, ichimoku, macd_hist, chaikin, stoch_rsi,
        parallel_sar_strategy, adx_trend, cci_signal, roc, ultimate_oscillator,
        pivot_points, heikin_ashi, renko_trend, demark, klinger,
    )
    fn_map = {
        "turtle": turtle.turtle_signals,
        "rsi": rsi_strategy.rsi_signal,
        "bollinger": bollinger.bollinger_signal,
        "grid": grid.grid_signal,
        "zscore": mean_reversion_z.zscore_signal,
        "donchian": donchian.donchian_signal,
        "keltner": keltner.keltner_signal,
        "coppock": coppock.coppock_signal,
        "vol_breakout": vol_breakout.vol_breakout_signal,
        "rsi_divergence": rsi_divergence.rsi_divergence,
        "sar": parallel_sar.sar_signal,
        "ou": ou_mean_reversion.ou_signal,
        "ml": ml_signal.ml_signal,
        "adaptive": adaptive.adaptive_position,
        "seasonal": seasonal.month_signal,
        # 迭代 111-120
        "supertrend": supertrend.supertrend_signal,
        "aroon": aroon.aroon_signal,
        "vortex": vortex.vortex_signal,
        "trix": trix_strategy.trix_signal,
        "williams_r": williams_r.williams_r_signal,
        "elder_ray": elder_ray.elder_ray_signal,
        "ichimoku": ichimoku.ichimoku_signal,
        "macd_hist": macd_hist.macd_hist_signal,
        "chaikin": chaikin.chaikin_signal,
        "stoch_rsi": stoch_rsi.stoch_rsi_signal,
        # 迭代 131-135
        "parabolic_sar": parallel_sar_strategy.parabolic_sar_signal,
        "adx_trend": adx_trend.adx_trend_signal,
        "cci_signal": cci_signal.cci_signal,
        "roc": roc.roc_signal,
        "ultimate_oscillator": ultimate_oscillator.ultimate_oscillator_signal,
        # 迭代 141-145
        "pivot_points": pivot_points.pivot_points_signal,
        "heikin_ashi": heikin_ashi.heikin_ashi_signal,
        "renko_trend": renko_trend.renko_trend_signal,
        "demark": demark.demark_signal,
        "klinger": klinger.klinger_signal,
    }
    for name, (kind, desc) in _FN_SPECS.items():
        fn = fn_map[name]
        reg[name] = lambda fn=fn, kind=kind, desc=desc: FnStrategy(fn, kind, desc)
        info[name] = desc
    return reg, info


_FN_REGISTRY, _FN_INFO = _build_fn_registry()

# 合并：旧类式策略 + 新函数式策略
STRATEGY_REGISTRY = {}
STRATEGY_REGISTRY.update(EXAMPLES_REGISTRY)          # sma_cross / macd / momentum / breakout / mean_reversion
STRATEGY_REGISTRY.update(_FN_REGISTRY)               # 迭代31-100 新策略

STRATEGY_INFO = {}
for _n, _c in EXAMPLES_REGISTRY.items():
    STRATEGY_INFO[_n] = getattr(_c, "description", _n)
STRATEGY_INFO.update(_FN_INFO)

STRATEGY_NAMES = list(STRATEGY_REGISTRY.keys())


def get_strategy(name: str, **kwargs) -> StrategyBase:
    """统一策略工厂。name 命中旧类式或新函数式皆可。"""
    if name in EXAMPLES_REGISTRY:
        return _get_examples(name, **kwargs)
    if name in _FN_REGISTRY:
        return _FN_REGISTRY[name](**kwargs)
    raise ValueError(f"未知策略: {name}，可选: {STRATEGY_NAMES}")
