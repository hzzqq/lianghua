"""非有限值（NaN / ±inf）判定的共享谓词。

**为什么需要它**：Python 的朴素校验在 NaN 面前全部静默失效——
``NaN or 0.0`` 得到的还是 NaN（NaN 是真值）、``NaN <= 0`` / ``NaN > x``
恒为 False。在一个真金白银的量化系统里，这意味着风控闸门被放行、止损判定
恒不触发，而且**一句报错都没有**。

**边界很明确**：本模块只提供*谓词*，不提供"安全网"。判定坏值之后该拒单、
该回退、还是该抛错，由每个调用方按自己的语义各自决定并各自守卫——安全网
集中到一处就等于只剩一处可被绕过。
"""
from __future__ import annotations

import math

__all__ = ["is_finite_num", "positive_finite"]


def is_finite_num(x) -> bool:
    """x 能否转成有限浮点数（None / 非数字 / NaN / ±inf 一律为 False）。"""
    if x is None or isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def positive_finite(x) -> float | None:
    """把 x 规整为"可用于交易决策的正数"，否则返回 None。

    返回值要么是有限正数、要么是 None，因此 ``positive_finite(a) or fallback``
    这种写法是安全的——这正是朴素 ``a or fallback`` 做不到的。
    """
    if not is_finite_num(x):
        return None
    v = float(x)
    return v if v > 0 else None
