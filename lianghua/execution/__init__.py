"""执行层：模拟与实盘 broker 统一接口。

- SimBroker  : 本地模拟（默认，离线可用）
- QmtBroker  : 迅投 QMT 实盘适配器（需 MiniQMT）
- PtBroker   : 恒生 PTrade 实盘适配器（需客户端）
通过 make_broker(kind, **cfg) 在模拟与真实柜台间切换。
"""
from __future__ import annotations

from .brokers import (
    BaseBroker, Order, Position, SimBroker, QmtBroker, PtBroker,
    make_broker, REGISTRY,
)

__all__ = ["BaseBroker", "Order", "Position", "SimBroker", "QmtBroker",
           "PtBroker", "make_broker", "REGISTRY"]
