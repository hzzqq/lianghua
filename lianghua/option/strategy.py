"""期权策略：基于标的与希腊字母生成交易信号 + 期权组合策略损益（迭代25）。

信号部分（作用于期权行情 DataFrame）：
- 1 = 买入期权(支付权利金/做多波动率)，-1 = 平仓/卖出，0 = 持有。

组合部分（迭代25 新增，纯损益计算，不依赖行情）：
- vertical_spread      垂直价差（牛市/熊市）
- iron_condor          铁鹰
- butterfly            蝶式
- payoff_curve         给定组合在到期日的损益曲线
- combo_payoff         组合损益（含净权利金）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "OptionStrategyBase", "CallTrendStrategy", "PutHedgeStrategy",
    "StraddleStrategy", "REGISTRY", "get_option_strategy",
    # 迭代25 新增
    "Leg", "vertical_spread", "iron_condor", "butterfly",
    "payoff_curve", "combo_payoff",
    # 迭代 181+ 超目标新增
    "stock_leg", "covered_call", "protective_put", "collar", "straddle",
    "strangle", "iron_butterfly", "calendar_spread", "ratio_spread",
    "diagonal_spread", "long_call", "long_put",
    "OPTION_COMBO_REGISTRY", "get_option_combo",
    # 迭代 44 新增
    "analyze_combo",
]


class OptionStrategyBase:
    """期权策略基类。"""

    name: str = "option_base"
    description: str = "期权策略基类"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class CallTrendStrategy(OptionStrategyBase):
    """看涨趋势：标的中期动量向上时买入认购(CALL)，向下则平仓。"""

    name = "option_call_trend"
    description = "标的动量向上买 CALL"

    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mom = df["underlying"].pct_change(self.window)
        sig = pd.Series(0, index=df.index)
        sig[mom > 0] = 1
        sig[mom < 0] = -1
        return sig.diff().fillna(0).clip(-1, 1)


class PutHedgeStrategy(OptionStrategyBase):
    """保险策略：标的下跌趋势时买入认沽(PUT)做对冲，回升平仓。"""

    name = "option_put_hedge"
    description = "下跌买 PUT 对冲"

    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mom = df["underlying"].pct_change(self.window)
        sig = pd.Series(0, index=df.index)
        sig[mom < 0] = 1   # 下跌时持有 PUT（注：期权 df 已按 PUT 合约构造）
        sig[mom > 0] = -1
        return sig.diff().fillna(0).clip(-1, 1)


class StraddleStrategy(OptionStrategyBase):
    """跨式突破：标的短期波动放大（突破）时买入跨式，回归时平仓。"""

    name = "option_straddle"
    description = "波动率突破买跨式"

    def __init__(self, lookback: int = 20, thr: float = 0.02):
        self.lookback = lookback
        self.thr = thr

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ret = df["underlying"].pct_change()
        roll_std = ret.rolling(self.lookback).std()
        sig = pd.Series(0, index=df.index)
        # 波动放大且出现大单日波动 -> 建仓
        sig[(roll_std > roll_std.median()) & (ret.abs() > self.thr)] = 1
        sig[(roll_std < roll_std.median()) & (ret.abs() < self.thr)] = -1
        return sig.diff().fillna(0).clip(-1, 1)


REGISTRY = {
    "option_call_trend": CallTrendStrategy,
    "option_put_hedge": PutHedgeStrategy,
    "option_straddle": StraddleStrategy,
}


def get_option_strategy(name: str, **kwargs) -> OptionStrategyBase:
    if name not in REGISTRY:
        raise ValueError(f"未知期权策略: {name}，可选: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


# ============================================================
# 迭代25：期权组合策略（纯损益计算）
# ============================================================

_VALID_SIDES = {"buy", "sell"}
_VALID_OTYPES = {"CALL", "PUT", "STOCK"}
_VALID_KINDS = {"option", "stock"}


class Leg:
    """单个期权腿。side: buy/sell；otype: CALL/PUT。"""

    def __init__(self, side: str, otype: str, strike: float, premium: float, kind: str = "option"):
        side = str(side).lower()
        otype = str(otype).upper()
        kind = str(kind).lower()
        if side not in _VALID_SIDES:
            raise ValueError(f"无效 side: {side}，应为 buy/sell")
        if kind == "stock":
            otype = "STOCK"
        if otype not in _VALID_OTYPES:
            raise ValueError(f"无效 otype: {otype}，应为 CALL/PUT/STOCK")
        if kind not in _VALID_KINDS:
            raise ValueError(f"无效 kind: {kind}，应为 option/stock")
        strike = float(strike)
        premium = float(premium)
        if not np.isfinite(strike) or strike < 0:
            raise ValueError(f"行权价必须 >=0 且有限: {strike}")
        if not np.isfinite(premium) or premium < 0:
            raise ValueError(f"权利金必须 >=0 且有限: {premium}")
        self.side = side                     # buy / sell
        self.otype = otype                   # CALL / PUT / STOCK
        self.strike = strike
        self.premium = premium
        self.kind = kind                     # option / stock

    def intrinsic(self, S: float) -> float:
        if self.kind == "stock":
            return S if self.side == "buy" else -S
        if self.otype == "CALL":
            return max(S - self.strike, 0.0)
        return max(self.strike - S, 0.0)

    def pnl(self, S: float) -> float:
        # 买方：到期内在价值 - 权利金；卖方：权利金 - 到期内在价值
        if self.side == "buy":
            return self.intrinsic(S) - self.premium
        return self.premium - self.intrinsic(S)

    def __repr__(self):
        return f"{self.side.upper()} {self.otype} K={self.strike} P={self.premium}"


def combo_payoff(legs: list[Leg], S: float) -> float:
    """组合在标的价格 S 处的到期损益（已计入净权利金）。"""
    if not legs:
        raise ValueError("combo_payoff 至少需要一条腿（legs 为空）")
    return float(sum(leg.pnl(S) for leg in legs))


def payoff_curve(legs: list[Leg], S_lo: float, S_hi: float, n: int = 200) -> pd.DataFrame:
    """生成组合到期损益曲线（DataFrame: S, pnl）。"""
    if not legs:
        raise ValueError("payoff_curve 至少需要一条腿（legs 为空）")
    if not np.isfinite(S_lo) or not np.isfinite(S_hi):
        raise ValueError("S_lo/S_hi 必须为有限数")
    if S_hi <= S_lo:
        raise ValueError("S_hi 必须大于 S_lo")
    n = max(int(n), 2)
    grid = np.linspace(S_lo, S_hi, n)
    pnls = [combo_payoff(legs, s) for s in grid]
    return pd.DataFrame({"S": grid, "pnl": pnls})


def analyze_combo(legs: list[Leg], S_lo: float, S_hi: float, n: int = 400) -> dict:
    """分析组合到期损益结构：最大盈利、最大亏损、盈亏平衡点、净权利金。

    新增能力（此前模块只能构造组合、无法解释其风险收益结构）。
    - max_profit / max_loss：损益曲线上的极值（inf 表示无界，如裸卖期权）。
    - breakevens：pnl 由负转正/由正转负的标的价格点列表。
    - net_premium：建仓净权利金（正数=收租，负数=净支出）。
    """
    if not legs:
        raise ValueError("analyze_combo 至少需要一条腿（legs 为空）")
    curve = payoff_curve(legs, S_lo, S_hi, n=n)
    pnls = curve["pnl"].to_numpy()
    S = curve["S"].to_numpy()
    finite = np.isfinite(pnls)
    max_profit = float(np.max(pnls[finite])) if finite.any() else float("inf")
    max_loss = float(np.min(pnls[finite])) if finite.any() else float("-inf")
    # 盈亏平衡点：pnl 穿越 0 的位置（线性插值）
    breakevens = []
    sign = np.sign(pnls)
    cross = np.where((sign[:-1] * sign[1:]) < 0)[0]
    for i in cross:
        s0, s1 = S[i], S[i + 1]
        p0, p1 = pnls[i], pnls[i + 1]
        if p1 == p0:
            continue
        b = s0 - p0 * (s1 - s0) / (p1 - p0)
        breakevens.append(float(b))
    if not breakevens and finite.any():
        # 区间内单边不穿越，但若端点在 0 附近也算
        if abs(pnls[0]) < 1e-9:
            breakevens.append(float(S[0]))
        if abs(pnls[-1]) < 1e-9:
            breakevens.append(float(S[-1]))
    return {
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakevens": sorted(breakevens),
        "net_premium": _prem(legs),
        "n_legs": len(legs),
    }


def _prem(legs: list[Leg]) -> float:
    # 现金流约定：买入腿是现金流出（净支出，记负），卖出腿是现金流入（收租，记正）。
    # 即 net_premium 为负 = 建仓净支付；为正 = 建仓净收取（收租）。
    return sum((-leg.premium if leg.side == "buy" else leg.premium) for leg in legs)


def vertical_spread(otype: str, low: float, high: float,
                    premium_low: float, premium_high: float,
                    direction: str = "bull") -> list[Leg]:
    """垂直价差。direction=bull 牛市看涨价差（买低卖高），bear 熊市看跌价差。"""
    otype = otype.upper()
    if direction == "bull":
        # 牛市：买低行权价，卖高行权价
        return [Leg("buy", otype, low, premium_low),
                Leg("sell", otype, high, premium_high)]
    else:
        # 熊市：卖低行权价，买高行权价
        return [Leg("sell", otype, low, premium_low),
                Leg("buy", otype, high, premium_high)]


def iron_condor(put_k_low: float, put_k_high: float,
               call_k_low: float, call_k_high: float,
               prem_put_low: float, prem_put_high: float,
               prem_call_low: float, prem_call_high: float) -> list[Leg]:
    """铁鹰：卖出看跌价差 + 卖出看涨价差，赚取时间价值/低波动。"""
    return [
        Leg("buy", "PUT", put_k_low, prem_put_low),
        Leg("sell", "PUT", put_k_high, prem_put_high),
        Leg("sell", "CALL", call_k_low, prem_call_low),
        Leg("buy", "CALL", call_k_high, prem_call_high),
    ]


def butterfly(otype: str, low: float, mid: float, high: float,
              prem_low: float, prem_mid: float, prem_high: float) -> list[Leg]:
    """蝶式：买入外侧两腿 + 卖出中间两腿（标准蝶式 1:2:1）。"""
    otype = otype.upper()
    return [
        Leg("buy", otype, low, prem_low),
        Leg("sell", otype, mid, prem_mid),
        Leg("sell", otype, mid, prem_mid),
        Leg("buy", otype, high, prem_high),
    ]


def stock_leg(side: str, price: float = 0.0) -> Leg:
    """股票腿：线性损益（无行权价），用于备兑/保护/领口等股票+期权组合。"""
    return Leg(side, "STOCK", 0.0, float(price), kind="stock")


# ============================================================
# 迭代 181+：更多期权组合策略（超自驱动目标，注册表驱动）
# ============================================================
def covered_call(spot: float, call_k: float, call_prem: float) -> list[Leg]:
    """备兑看涨：持有股票 + 卖 CALL，用权利金增强收益、封顶上行。"""
    return [stock_leg("buy", spot), Leg("sell", "CALL", call_k, call_prem)]


def protective_put(spot: float, put_k: float, put_prem: float) -> list[Leg]:
    """保护性看跌：持有股票 + 买 PUT，付出权利金换取下行保险。"""
    return [stock_leg("buy", spot), Leg("buy", "PUT", put_k, put_prem)]


def collar(spot: float, put_k: float, call_k: float,
           put_prem: float, call_prem: float) -> list[Leg]:
    """领口：持有股票 + 买 PUT（保护） + 卖 CALL（补贴保费），限定收益区间。"""
    return [stock_leg("buy", spot), Leg("buy", "PUT", put_k, put_prem),
            Leg("sell", "CALL", call_k, call_prem)]


def straddle(otype: str, strike: float, call_prem: float, put_prem: float) -> list[Leg]:
    """跨式：同时买 CALL+PUT 同行权价，押注大幅波动（多空双买）。"""
    otype = otype.upper()
    return [Leg("buy", otype, strike, call_prem), Leg("buy", otype, strike, put_prem)]


def strangle(call_k: float, put_k: float, call_prem: float, put_prem: float) -> list[Leg]:
    """宽跨式：买虚值 CALL + 虚值 PUT，成本低于跨式，需更大波动才盈利。"""
    return [Leg("buy", "CALL", call_k, call_prem), Leg("buy", "PUT", put_k, put_prem)]


def iron_butterfly(otype: str, body: float, wing: float,
                   prem_body: float, prem_wing: float) -> list[Leg]:
    """铁蝶：卖 2 张 ATM（CALL+PUT）+ 买两侧虚值，赚取时间价值、低波动盈利。"""
    otype = otype.upper()
    return [
        Leg("sell", otype, body, prem_body),
        Leg("sell", otype, body, prem_body),
        Leg("buy", otype, wing, prem_wing),
        Leg("buy", otype, wing, prem_wing),
    ]


def calendar_spread(otype: str, near_k: float, far_k: float,
                    near_prem: float, far_prem: float) -> list[Leg]:
    """日历价差：卖近月 + 买远月（同/近行权价），赚时间价值衰减差。"""
    otype = otype.upper()
    return [Leg("sell", otype, near_k, near_prem), Leg("buy", otype, far_k, far_prem)]


def ratio_spread(otype: str, low: float, high: float,
                 prem_low: float, prem_high: float, ratio: int = 2) -> list[Leg]:
    """比率价差：买 1 张低行权价 + 卖 ratio 张高行权价，净权利金常为负（收租）。"""
    otype = otype.upper()
    legs = [Leg("buy", otype, low, prem_low)]
    legs += [Leg("sell", otype, high, prem_high) for _ in range(max(int(ratio), 1))]
    return legs


def diagonal_spread(otype: str, near_k: float, far_k: float,
                    near_prem: float, far_prem: float, direction: str = "bull") -> list[Leg]:
    """对角价差：卖近月 + 买远月、行权价不同，方向由行权价关系决定。"""
    otype = otype.upper()
    if direction == "bull":
        return [Leg("sell", otype, near_k, near_prem), Leg("buy", otype, far_k, far_prem)]
    return [Leg("buy", otype, near_k, near_prem), Leg("sell", otype, far_k, far_prem)]


def long_call(strike: float, premium: float) -> list[Leg]:
    """买入看涨：看大涨，损失有限（权利金）、上涨潜力不限。"""
    return [Leg("buy", "CALL", strike, premium)]


def long_put(strike: float, premium: float) -> list[Leg]:
    """买入看跌：看大跌/避险，损失有限（权利金）、下行潜力大。"""
    return [Leg("buy", "PUT", strike, premium)]


# 组合注册表：UI / 自驱动循环动态发现；defaults 提供合理默认参数用于一键预览。
OPTION_COMBO_REGISTRY = {
    "covered_call":   {"builder": covered_call, "desc": "备兑看涨（股票+卖CALL）",
                       "defaults": {"spot": 100.0, "call_k": 105.0, "call_prem": 2.0}},
    "protective_put": {"builder": protective_put, "desc": "保护性看跌（股票+买PUT）",
                       "defaults": {"spot": 100.0, "put_k": 95.0, "put_prem": 2.0}},
    "collar":         {"builder": collar, "desc": "领口（股票+买PUT+卖CALL）",
                       "defaults": {"spot": 100.0, "put_k": 95.0, "call_k": 105.0,
                                    "put_prem": 1.5, "call_prem": 1.5}},
    "straddle":       {"builder": straddle, "desc": "跨式（买CALL+买PUT同价）",
                       "defaults": {"otype": "CALL", "strike": 100.0,
                                    "call_prem": 3.0, "put_prem": 3.0}},
    "strangle":       {"builder": strangle, "desc": "宽跨式（买虚值CALL+PUT）",
                       "defaults": {"call_k": 110.0, "put_k": 90.0,
                                    "call_prem": 1.5, "put_prem": 1.5}},
    "iron_condor":    {"builder": iron_condor, "desc": "铁鹰（卖双价差）",
                       "defaults": {"put_k_low": 90.0, "put_k_high": 95.0,
                                    "call_k_low": 105.0, "call_k_high": 110.0,
                                    "prem_put_low": 0.8, "prem_put_high": 2.0,
                                    "prem_call_low": 2.0, "prem_call_high": 0.8}},
    "butterfly":      {"builder": butterfly, "desc": "蝶式（1:2:1）",
                       "defaults": {"otype": "CALL", "low": 90.0, "mid": 100.0,
                                    "high": 110.0, "prem_low": 1.0,
                                    "prem_mid": 4.0, "prem_high": 1.0}},
    "iron_butterfly": {"builder": iron_butterfly, "desc": "铁蝶（卖ATM+买虚值）",
                       "defaults": {"otype": "CALL", "body": 100.0, "wing": 110.0,
                                    "prem_body": 4.0, "prem_wing": 1.0}},
    "calendar_spread":{"builder": calendar_spread, "desc": "日历价差（卖近买远）",
                       "defaults": {"otype": "CALL", "near_k": 100.0, "far_k": 100.0,
                                    "near_prem": 2.0, "far_prem": 3.5}},
    "ratio_spread":   {"builder": ratio_spread, "desc": "比率价差（买1卖N）",
                       "defaults": {"otype": "CALL", "low": 100.0, "high": 110.0,
                                    "prem_low": 3.0, "prem_high": 1.5, "ratio": 2}},
    "diagonal_spread":{"builder": diagonal_spread, "desc": "对角价差（近卖远买）",
                       "defaults": {"otype": "CALL", "near_k": 100.0, "far_k": 105.0,
                                    "near_prem": 2.0, "far_prem": 3.5, "direction": "bull"}},
    "vertical_spread": {"builder": vertical_spread, "desc": "垂直价差（牛/熊）",
                        "defaults": {"otype": "CALL", "low": 100.0, "high": 110.0,
                                     "premium_low": 3.0, "premium_high": 1.0,
                                     "direction": "bull"}},
    "long_call":      {"builder": long_call, "desc": "买入看涨",
                       "defaults": {"strike": 100.0, "premium": 3.0}},
    "long_put":       {"builder": long_put, "desc": "买入看跌",
                       "defaults": {"strike": 100.0, "premium": 3.0}},
}


def get_option_combo(name: str, **kwargs) -> list[Leg]:
    """按名构建期权组合（legs）。未给参数时自动套用 defaults。"""
    if name not in OPTION_COMBO_REGISTRY:
        raise ValueError(f"未知期权组合: {name}，可选: {list(OPTION_COMBO_REGISTRY)}")
    spec = OPTION_COMBO_REGISTRY[name]
    kw = dict(spec["defaults"])
    kw.update(kwargs)
    return spec["builder"](**kw)
