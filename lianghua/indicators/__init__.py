"""技术指标库（纯 pandas/numpy）。"""
import inspect

from .tech import (rsi, macd, kdj, boll, atr, cci, obv, vwap,
                   stochastic, williams_r)
from .tech4 import (adx, pivot, heikin_ashi, renko, demarker, klinger,
                    chaikin_volatility, force_index, know_sure_thing,
                    zigzag, price_channels)
from . import tech, tech2, tech3, tech4

__all__ = ["rsi", "macd", "kdj", "boll", "atr", "cci", "obv", "vwap",
           "stochastic", "williams_r",
           "adx", "pivot", "heikin_ashi", "renko", "demarker", "klinger",
           "chaikin_volatility", "force_index", "know_sure_thing",
           "zigzag", "price_channels", "INDICATOR_FUNCS"]

# 统一函数映射：name -> callable，供 UI/CLI 动态发现与调用。
INDICATOR_FUNCS = {}
for _mod in (tech, tech2, tech3, tech4):
    for _name, _obj in inspect.getmembers(_mod, inspect.isfunction):
        if not _name.startswith("_"):
            INDICATOR_FUNCS.setdefault(_name, _obj)
