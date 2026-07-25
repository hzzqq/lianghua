"""执行层 broker 包：统一可插拔接口。

- SimBroker  : 本地模拟撮合（默认，离线可用）
- QmtBroker  : 迅投 QMT 实盘适配器（需 MiniQMT + xtquant）
- PtBroker   : 恒生 PTrade 实盘适配器（需 PtApi 环境）

所有 broker 实现同一 BaseBroker 接口，run_backtest / UI 可通过
`make_broker("qmt", **cfg)` 在模拟与真实柜台间无缝切换。
"""
from __future__ import annotations

from .base import BaseBroker, Order, Position
from .sim import SimBroker
from .paper import PaperBroker
from .qmt import QmtBroker
from .pt import PtBroker

REGISTRY = {
    "sim": SimBroker,
    "paper": PaperBroker,
    "qmt": QmtBroker,
    "pt": PtBroker,
}


def make_broker(kind: str = "sim", **kwargs):
    """工厂：按名称创建 broker。kind in {sim, qmt, pt}。"""
    if kind not in REGISTRY:
        raise ValueError(f"未知 broker 类型: {kind}，可选 {list(REGISTRY)}")
    return REGISTRY[kind](**kwargs)


__all__ = ["BaseBroker", "Order", "Position", "SimBroker", "PaperBroker",
           "QmtBroker", "PtBroker", "REGISTRY", "make_broker"]
