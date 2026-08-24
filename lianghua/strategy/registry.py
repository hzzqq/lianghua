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
# 三要素： (kind, 中文名, 一句话说明, 详细描述)
#   - 中文名：UI 下拉直接展示，避免英文 key 暴露给用户
#   - 一句话说明：STRATEGY_INFO（卡片副标题/tooltip）
#   - 详细描述：STRATEGY_DETAIL（逻辑/适用场景/参数/风险）
_FN_SPECS = {
    "turtle":        ("df", "海龟通道突破", "海龟通道突破（唐奇安+ATR）",
        "经典的「海龟交易法则」：以 N 日（默认 20/55）唐奇安通道突破为入场信号，"
        "ATR 度量波动并据此计算头寸规模与止损距离。偏向中长线趋势跟踪，"
        "在单边市中表现好，震荡市频繁假突破。参数：入场通道 20/55、ATR 周期 20、止损 2*ATR。"),
    "rsi":           ("df", "RSI 超买超卖", "RSI 超买超卖（30/70）",
        "相对强弱指标 RSI 在 0–100 区间波动，通常以 70 为超买、30 为超卖阈值。"
        "价格在超卖区拐头向上时买入，超买区拐头向下时卖出。适合震荡市反转，"
        "强趋势中会出现「超买再超买」钝化，需结合趋势过滤。参数：RSI 周期 14、阈值 30/70。"),
    "bollinger":     ("df", "布林带均值回复", "布林带均值回复",
        "由中轨（N 日均线的）、上下轨（±K 倍标准差）构成通道。价格触及下轨视为超卖买入、"
        "触及上轨视为超买卖出，赚取均值回复收益。适合区间震荡；突破行情中通道会张口，"
        "需配合带宽判断。参数：均线 20、标准差倍数 2。"),
    "grid":          ("grid", "网格交易", "网格交易（档位符号化为方向）",
        "在基准价上下按固定/等比间距布设多档买卖单：跌到下一档买入、涨回上一档卖出，"
        "在震荡行情中反复收割价差。优点是震荡市收益稳定，缺点是单边下跌会不断满仓、"
        "单边上涨会踏空。参数：网格数、间距比例、基准价。"),
    "zscore":        ("df", "Z-score 均值回归", "Z-score 均值回归",
        "计算价格（或价差值）相对其滚动均值的标准差倍数（Z 分数）。Z 极低（如 < -2）"
        "视为超卖买入，Z 极高（如 > 2）视为超卖/卖空。基于「价格围绕均值波动」假设，"
        "适合配对/价差类平稳序列，趋势市失效。参数：窗口 20、阈值 ±2。"),
    "donchian":      ("close", "唐奇安通道突破", "唐奇安通道突破",
        "取 N 日最高价与最低价构成通道，价格突破上轨做多、跌破下轨做空。"
        "极简的趋势跟踪系统（海龟同源），对参数敏感，长周期更平滑。参数：通道 20/55。"),
    "keltner":       ("hlc", "Keltner 通道突破", "Keltner 通道突破",
        "以 EMA 为中轨、±N 倍 ATR 为带宽的通道。价格突破上轨做多、跌破下轨做空，"
        "相比布林带对波动更稳健（ATR 而非标准差）。适合趋势跟踪与突破确认。参数：EMA 20、ATR 10、倍数 2。"),
    "coppock":       ("close", "考普克长短周期动量", "考普克长短周期动量",
        "考普克曲线 = 长周期（11/14 月）动量经 10 月 EMA 平滑，原本用于判断市场底部。"
        "本实现转为中长线动量择时：曲线上穿 0 看多、下穿 0 看空。适合大类/指数中长线，滞后明显。"),
    "vol_breakout":  ("close", "波动率突破", "波动率突破",
        "以近期波动率（如 ATR 或标准差）的倍数作为突破阈值：当日波动超过阈值才触发交易，"
        "过滤小幅噪音。适合事件驱动/跳空行情，避免在无序波动中频繁进出。参数：波动率窗口、倍数。"),
    "rsi_divergence":("close_rsi", "RSI 顶底背离", "RSI 顶/底背离",
        "价格创新高但 RSI 未创新高（顶背离）视为上涨乏力卖出；价格创新低但 RSI 未创新低"
        "（底背离）视为下跌衰竭买入。是反转预警信号，需等待确认 K 线再动手。参数：RSI 周期 14。"),
    "sar":           ("hl", "抛物线 SAR 趋势", "抛物线 SAR 趋势",
        "抛物线转向指标 SAR 随趋势加速贴近价格：价格在 SAR 上方为多头持有、下方为空头持有，"
        "反向穿越即反转。自带加速因子，趋势市跟得住、震荡市反复打脸。参数：初始 0.02、最大 0.2。"),
    "ou":            ("close", "OU 均值回复", "OU 均值回复",
        "基于 Ornstein–Uhlenbeck 过程的均值回复模型，估计价格向其长期均值的回归速度与均衡位。"
        "偏离均衡过大时反向建仓。适合平稳/均值回复资产（价差、利率、部分商品），趋势资产失效。"),
    "ml":            ("df", "机器学习择时", "机器学习择时（LR，失败降级动量）",
        "用逻辑回归等轻量模型，以多技术指标为特征预测下期涨跌方向。模型不可用时自动降级为"
        "动量信号以保证可用性。需防过拟合、注意样本外衰减。参数：特征集、训练窗口。"),
    "adaptive":      ("ret", "自适应波动仓位", "自适应波动仓位",
        "根据近期波动率动态调整仓位：波动大则降仓、波动小则加仓，目标是把组合波动稳定在目标水平"
        "（波动率目标化）。属于仓位管理而非方向信号，常与趋势/动量叠加。参数：目标波动、回望窗口。"),
    "seasonal":      ("close", "季节性月份效应", "季节性月份效应",
        "利用历史统计上的月份效应（如「五月卖出」「年初效应」）做择时。实现为按月份给出多/空偏向。"
        "本质是统计规律，近年有效性减弱，宜作辅助参考而非单独信号。参数：历史月份收益统计。"),
    # 迭代 111-120：进阶技术策略
    "supertrend":    ("df", "超级趋势", "Supertrend 趋势跟踪（ATR）",
        "基于 ATR 构造上下轨，价格在其上方显示看涨、下方显示看跌，并随趋势翻转。比 SAR 更平滑，"
        "是当下流行的趋势跟踪指标，适合日内与日线趋势。参数：ATR 10、倍数 3。"),
    "aroon":         ("df", "阿隆趋势强度", "Aroon 趋势强度",
        "Aroon Up/Down 衡量「距离 N 日内最高/最低已过去多久」，逼近 100 表示强势趋势、逼近 0 表示"
        "趋势疲弱。两者差值可判断牛熊转换。适合趋势强度判定与突破过滤。参数：周期 25。"),
    "vortex":        ("df", "涡旋指标", "Vortex 指标交叉",
        "Vortex VI+/VI- 分别捕捉向上/向下趋势涡旋，二者金叉/死叉作为趋势转向信号。"
        "对价格极值敏感，适合趋势确认，参数：周期 14。"),
    "trix":          ("close", "三重指数平滑动量", "TRIX 三重平滑动量交叉",
        "对收盘价做三次 EMA 后再取一阶差分得到 TRIX，穿越其信号线（MA）产生买卖。能滤除短期噪音、"
        "反映中长期动量，滞后较大。参数：EMA 15、信号 9。"),
    "williams_r":    ("df", "威廉指标", "Williams %R 超买超卖反转",
        "Williams %R 在 -100~0 区间，低于 -80 超卖、高于 -20 超买，用于捕捉反转。与 RSI 思路类似"
        "但刻度相反，适合震荡市抄底摸顶。参数：周期 14。"),
    "elder_ray":     ("df", "艾尔德多空力量", "Elder Ray 多空力量",
        "由 Bull Power（最高价-EMA）与 Bear Power（最低价-EMA）构成，衡量多空推动力量。"
        "两者与价格/EMA 的背离揭示趋势衰竭。适合趋势强度与转折辅助。参数：EMA 13。"),
    "ichimoku":      ("df", "一目均衡表", "一目均衡表云图",
        "包含转换线、基准线、迟行线及云带（前瞻上下轨）的完整趋势系统，云带上方偏多、下方偏空，"
        "多线交叉提供多层次信号。信息量大、适合趋势判断。参数：9/26/52。"),
    "macd_hist":     ("close", "MACD 柱穿越", "MACD 柱状图零轴穿越",
        "取 MACD 柱状图（DIF-DEA）穿越零轴作为多空信号，比直接用 DIF/DEA 交叉更早捕捉动能变化。"
        "适合中线趋势，配合量能更佳。参数：12/26/9。"),
    "chaikin":       ("df", "蔡金资金流", "Chaikin 资金流(CMF)",
        "Chaikin Money Flow 用「收盘价在当日区间的位置 × 成交量」衡量资金流入流出。CMF>0 吸筹、"
        "<0 派发，用于验证价格趋势的资金面。参数：周期 20。"),
    "stoch_rsi":     ("close", "随机RSI", "Stochastic RSI 超买超卖",
        "先算 RSI 再对 RSI 自身做随机指标，放大超买超卖信号、比 RSI 更灵敏。适合捕捉极端摆动，"
        "但假信号也更多，需过滤。参数：RSI 14、随机 14/3。"),
    # 迭代 131-135：进阶技术策略
    "parabolic_sar": ("df", "抛物线 SAR（进阶）", "抛物线 SAR 趋势",
        "同「抛物线 SAR 趋势」，进阶封装版（独立模块），参数与行为一致，便于与其他策略并列调用。"),
    "adx_trend":     ("df", "ADX 趋势强度顺势", "ADX 趋势强度(顺势)",
        "用 ADX 衡量趋势强度：ADX 高（如 >25）表示趋势明确，此时顺 +DI/-DI 方向交易；ADX 低则"
        "离场观望。解决「趋势/震荡识别」问题，常与方向指标搭配。参数：ADX 14。"),
    "cci_signal":    ("df", "CCI 商品通道", "CCI 均值回复",
        "CCI 衡量价格相对其统计均值（典型价均值 ±0.015 倍均值绝对偏差）的偏离。±100 为超买/超卖"
        "阈值，可用于均值回复或突破跟随。对异常值敏感，适合商品/周期股。参数：周期 20。"),
    "roc":           ("close", "变动率动量", "ROC 变动率动量",
        "Rate of Change 取 N 日价格变化百分比，反映动量强弱。ROC 上穿 0 看多、下穿 0 看空，"
        "或用于判断加速/减速。简单直观，适合动量系统与背离。参数：周期 12。"),
    "ultimate_oscillator": ("df", "终极波动指标", "终极波动指标(超买超卖)",
        "综合短/中/长三周期动量加权，缓解单一周期波动，在超买/超卖区出现背离时给出信号。"
        "比单一 RSI 更平滑，适合中短线反转。参数：7/14/28。"),
    # 迭代 141-145：进阶技术策略
    "pivot_points":  ("df", "轴心点", "轴心点(Pivot)多空分界",
        "用前一日高低收计算枢轴点 P 及支撑/阻力位（R1/S1…），价格站上 P 偏多、跌破偏空，"
        "是日内交易常用参考。参数：经典/斐波那契等多种算法。"),
    "heikin_ashi":   ("df", "平均K线", "Heikin-Ashi 云图趋势",
        "Heikin-Ashi 对 K 线做平均化处理，滤除噪音、让趋势更连贯，红绿长实体代表趋势、"
        "小十字代表转折。适合趋势跟踪与持仓，但滞后于真实价格。"),
    "renko_trend":   ("df", "砖形图趋势", "Renko 砖形趋势跟踪",
        "Renko 仅在价格变动超过固定砖块大小才画新砖，忽略时间、突出价格位移，趋势流畅、"
        "噪音少。适合趋势跟踪，但砖块大小选择影响灵敏度。参数：砖块大小。"),
    "demark":        ("df", "迪马克比较动量", "DeMark 比较动量",
        "Tom DeMark 系列（如 TD 序列/比较）：比较收盘价与前若干根收盘的关系生成_setup/倒计时，"
        "用于捕捉趋势 exhaustion。逻辑较繁，适合顶底转折。参数：比较长度 4/9。"),
    "klinger":       ("df", "克林格量价振荡", "Klinger 量价振荡器",
        "Klinger Oscillator 综合价量关系（以趋势体积原则）判断资金流向，信号线金叉/死叉作为买卖。"
        "兼顾量能与价格，适合中长线。参数：34/55。"),
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
    cn_map = {}
    detail_map = {}
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
    for name, (kind, cn, short, detail) in _FN_SPECS.items():
        fn = fn_map[name]
        reg[name] = lambda fn=fn, kind=kind, desc=short: FnStrategy(fn, kind, desc)
        info[name] = short
        cn_map[name] = cn
        detail_map[name] = detail
    return reg, info, cn_map, detail_map


_FN_REGISTRY, _FN_INFO, _FN_CN, _FN_DETAIL = _build_fn_registry()

# 合并：旧类式策略 + 新函数式策略
STRATEGY_REGISTRY = {}
STRATEGY_REGISTRY.update(EXAMPLES_REGISTRY)          # sma_cross / macd / momentum / breakout / mean_reversion
STRATEGY_REGISTRY.update(_FN_REGISTRY)               # 迭代31-100 新策略

STRATEGY_INFO = {}
STRATEGY_CN = {}
STRATEGY_DETAIL = {}
for _n, _c in EXAMPLES_REGISTRY.items():
    STRATEGY_INFO[_n] = getattr(_c, "description", _n)
    STRATEGY_CN[_n] = getattr(_c, "cn_name", _n)
    STRATEGY_DETAIL[_n] = getattr(_c, "detail", STRATEGY_INFO[_n])
STRATEGY_INFO.update(_FN_INFO)
STRATEGY_CN.update(_FN_CN)
STRATEGY_DETAIL.update(_FN_DETAIL)

STRATEGY_NAMES = list(STRATEGY_REGISTRY.keys())


def get_strategy(name: str, **kwargs) -> StrategyBase:
    """统一策略工厂。name 命中旧类式或新函数式皆可。"""
    if name in EXAMPLES_REGISTRY:
        return _get_examples(name, **kwargs)
    if name in _FN_REGISTRY:
        return _FN_REGISTRY[name](**kwargs)
    raise ValueError(f"未知策略: {name}，可选: {STRATEGY_NAMES}")
