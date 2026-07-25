"""期权定价与希腊字母（Black-Scholes 模型 + 二叉树美式期权）。

纯 NumPy 实现，零重依赖。覆盖看涨/看跌欧式与美式期权。
"""
from __future__ import annotations

import math
import numpy as np
from math import log, sqrt, erf


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return np.exp(-0.5 * x * x) / sqrt(2.0 * np.pi)


_VALID_OPT = {"CALL", "PUT"}


def _validate_opt_type(opt_type: str) -> str:
    """归一化并返回期权类型；非法类型显式抛错而非静默降级为 PUT。"""
    key = str(opt_type).upper()
    if key not in _VALID_OPT:
        raise ValueError(f"opt_type 必须是 CALL/PUT，收到: {opt_type!r}")
    return key


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "CALL") -> float:
    """Black-Scholes 欧式期权理论价格。

    S=标的价格, K=行权价, T=剩余期限(年), r=无风险利率, sigma=波动率。
    opt_type: CALL / PUT。

    退化的输入（T<=0 / sigma<=0 / S<=0 / K<=0）按内在价值处理。
    """
    otype = _validate_opt_type(opt_type)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # 到期或退化情形：按内在价值
        intrinsic = max(S - K, 0.0) if otype == "CALL" else max(K - S, 0.0)
        return float(intrinsic)
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if otype == "CALL":
        return float(S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2))
    else:
        return float(K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1))


def greeks(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "CALL") -> dict:
    """返回 Delta / Gamma / Vega / Theta / Rho。"""
    otype = _validate_opt_type(opt_type)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf = _norm_pdf(d1)
    gamma = pdf / (S * sigma * sqrtT)
    vega = S * pdf * sqrtT  # 每 1.00 波动率变动；常用每 1% 需 /100
    is_call = otype == "CALL"
    delta = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0
    if is_call:
        theta = -(S * pdf * sigma) / (2 * sqrtT) - r * K * np.exp(-r * T) * _norm_cdf(d2)
        rho = K * T * np.exp(-r * T) * _norm_cdf(d2)
    else:
        theta = -(S * pdf * sigma) / (2 * sqrtT) + r * K * np.exp(-r * T) * _norm_cdf(-d2)
        rho = -K * T * np.exp(-r * T) * _norm_cdf(-d2)
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega / 100.0),   # 转为每 1% 波动率
        "theta": float(theta / 365.0),  # 转为每日 theta
        "rho": float(rho / 100.0),
    }


def implied_vol(price: float, S: float, K: float, T: float, r: float, opt_type: str = "CALL",
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-6) -> float:
    """二分法反解隐含波动率。

    新增强化：
    - 校验 opt_type（非法显式抛错）。
    - 校验输入合法性（S/K/T 必须为正；否则无解返回 NaN）。
    - 检测目标价是否落在无套利区间内：若不在 [bs_price(lo), bs_price(hi)]
      范围内，说明报价无意义，返回 NaN 而非静默给出一个错误波动率。
    - 收敛后以两侧边界检查：若残差仍超过 tol，返回 NaN 表示未收敛。
    """
    otype = _validate_opt_type(opt_type)
    if not (math.isfinite(price) and math.isfinite(S) and math.isfinite(K)
            and math.isfinite(T) and math.isfinite(r)):
        return float("nan")
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    intrinsic = max(S - K, 0.0) if otype == "CALL" else max(K - S, 0.0)
    if price <= intrinsic:
        # 价格不超过内在价值：无正波动率解
        return 0.0

    price_lo = bs_price(S, K, T, r, lo, otype)
    price_hi = bs_price(S, K, T, r, hi, otype)
    if price < price_lo or price > price_hi:
        # 目标价落在无套利区间外：无法反解
        return float("nan")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = bs_price(S, K, T, r, mid, otype)
        if val == price:
            return float(mid)
        if abs(val - price) < tol:
            return float(mid)
        if val > price:
            hi = mid
        else:
            lo = mid
    # 未完全收敛：残差检查
    mid = 0.5 * (lo + hi)
    if abs(bs_price(S, K, T, r, mid, otype) - price) < tol * 10:
        return float(mid)
    return float("nan")


def binomial_price(S: float, K: float, T: float, r: float, sigma: float,
                   opt_type: str = "CALL", steps: int = 100, american: bool = True) -> float:
    """Cox-Ross-Rubinstein 二叉树期权定价（新增能力）。

    支持美式（默认）与欧式期权，弥补 Black-Scholes 只能给欧式定价的短板。
    steps 越大越精确。退化的输入按内在价值处理。

    隐性修复：steps<=0 直接按内在价值返回，避免 0 除与空网格崩溃。
    """
    otype = _validate_opt_type(opt_type)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or steps <= 0:
        intrinsic = max(S - K, 0.0) if otype == "CALL" else max(K - S, 0.0)
        return float(intrinsic)

    dt = T / steps
    disc = math.exp(-r * dt)
    if sigma <= 0 or dt <= 0:
        intrinsic = max(S - K, 0.0) if otype == "CALL" else max(K - S, 0.0)
        return float(intrinsic)
    u = math.exp(sigma * sqrt(dt))
    d = 1.0 / u
    pu = (math.exp(r * dt) - d) / (u - d)  # 风险中性上涨概率
    if not (0.0 <= pu <= 1.0):
        # 数值退化（极端参数）按内在价值兜底
        intrinsic = max(S - K, 0.0) if otype == "CALL" else max(K - S, 0.0)
        return float(intrinsic)
    pd = 1.0 - pu

    # 终端标的路径价格
    spot = np.array([S * (u ** i) * (d ** (steps - i)) for i in range(steps + 1)], dtype=float)
    if otype == "CALL":
        vals = np.maximum(spot - K, 0.0)
    else:
        vals = np.maximum(K - spot, 0.0)

    is_call = otype == "CALL"
    for step in range(steps - 1, -1, -1):
        # 回撤一期的标的路径
        spot = spot[:step + 1] * u  # 等价于上移一格（与下一节点对齐）
        cont = disc * (pu * vals[1:] + pd * vals[:-1])
        if american:
            if is_call:
                exercise = np.maximum(spot - K, 0.0)
            else:
                exercise = np.maximum(K - spot, 0.0)
            vals = np.maximum(cont, exercise)
        else:
            vals = cont

    return float(vals[0])
