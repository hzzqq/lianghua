"""多账户路由（第十三轮·深化方向三：多账户路由）。

把一组标的按规则路由到不同 broker 账户，每个账户可配置独立策略/资金/风控。
典型场景：真实柜台小资金实盘 + paper 大资金演练；股票户 vs 期货户分账。
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from .live import LiveEngine, make_live_engine
from .order_book import OrderBook


class AccountRouter:
    """路由规则：by_asset(资产类型) > by_prefix(代码前缀) > default。"""

    def __init__(self, routes=None, default: str = "default"):
        r = routes or {}
        self.by_asset = r.get("by_asset", {})
        self.by_prefix = r.get("by_prefix", {})
        self.default = r.get("default", default) or default
        # 校验路由值均为非空字符串，避免路由到 None/非法账户静默失效
        for scope, mapping in (("by_asset", self.by_asset),
                               ("by_prefix", self.by_prefix)):
            for key, acc in mapping.items():
                if not isinstance(acc, str) or not acc:
                    raise ValueError(f"路由 {scope}[{key!r}] 必须为非空账户名，收到 {acc!r}")

    def route(self, symbol, asset_type) -> str:
        atv = asset_type.value if hasattr(asset_type, "value") else str(asset_type)
        if atv in self.by_asset:
            return self.by_asset[atv]
        for pre, acc in self.by_prefix.items():
            if str(symbol).startswith(str(pre)):
                return acc
        return self.default

    def preview(self, symbols) -> dict:
        """批量路由预览（新增可观测能力）：返回 {symbol: account}。

        asset_type 由 symbol 自动推断（与 MultiLiveEngine._asset_of 同口径），
        便于在下单前确认每个标的会被分配到哪个账户，而非事后才发现错配。
        """
        from ..core.assets import detect_asset_type
        out: dict = {}
        for sym in symbols:
            out[sym] = self.route(sym, detect_asset_type(sym))
        return out

    def __repr__(self):
        return "AccountRouter(by_asset=%s, by_prefix=%s, default=%s)" % (
            self.by_asset, self.by_prefix, self.default)


class MultiLiveEngine:
    """多账户实盘引擎：每个账户一个 LiveEngine，按 router 分配标的后各自 step。"""

    def __init__(self, engines, router, symbols=None, notify=None,
                 order_db: str = "live_orders.db"):
        self.engines = dict(engines)
        self.router = router
        self.symbols = list(symbols or [])
        self.notify = notify
        self.book = OrderBook(order_db)
        self.killed = False
        self.last_status = None

    def _asset_of(self, sym):
        for eng in self.engines.values():
            try:
                return eng._asset(sym)
            except Exception:
                continue
        from ..core.assets import detect_asset_type
        return detect_asset_type(sym)

    def step(self):
        if self.killed:
            return {"killed": True, "accounts": {}}
        buckets = {name: [] for name in self.engines}
        for sym in self.symbols:
            acc = self.router.route(sym, self._asset_of(sym))
            buckets.setdefault(acc, []).append(sym)
        results = {}
        for name, eng in self.engines.items():
            syms = buckets.get(name, [])
            if not syms:
                results[name] = {"skipped": True, "reason": "无分配标的"}
                continue
            eng.symbols = syms
            try:
                results[name] = eng.step()
            except Exception as e:
                results[name] = {"error": str(e)}
        self.last_status = {"ts": str(_dt.datetime.now()),
                            "accounts": results, "killed": self.killed}
        return self.last_status

    def run(self, max_steps: int = 1, interval: float = 0.0):
        out = []
        for _ in range(int(max_steps)):
            if self.killed:
                break
            out.append(self.step())
            if interval > 0:
                import time
                time.sleep(interval)
        return out

    def kill(self):
        self.killed = True
        for eng in self.engines.values():
            eng.kill()
        return {"killed": True}

    def status(self):
        return getattr(self, "last_status",
                       {"accounts": {}, "killed": self.killed})

    def rebalance_status(self) -> dict | None:
        """返回最近一次 rebalance 的计划快照（新增可观测能力）。

        调用 rebalance() 前返回 None；便于外部在两次再平衡之间读取最新计划，
        而不必重复触发再平衡副作用（修改 paper/sim 账户 capital）。
        """
        return getattr(self, "_last_rebalance", None)

    def rebalance(self, target_weights: dict | None = None,
                  write_log: str = "rebalance_log.json") -> dict:
        """跨账户权益再平衡：把每账户 equity 拉向目标权重。

        - 真实柜台（qmt/pt）无法跨券商划账，故仅返回 transfer 计划，
          由人工/银证转账执行；
        - paper/sim 账户直接把 engine.capital 调成目标值，后续 step 自然再平衡。
        """
        rows: dict[str, dict] = {}
        total = 0.0
        for name, eng in self.engines.items():
            acct = eng.broker.get_account({})
            eq = float(acct.get("equity", acct.get("cash", 0.0)))
            broker_name = type(eng.broker).__name__
            is_real = broker_name in ("QmtBroker", "PtBroker")
            rows[name] = {"equity": eq, "engine": eng, "is_real": is_real}
            total += eq
        if target_weights is not None:
            if not isinstance(target_weights, dict):
                raise TypeError("target_weights 必须是 {账户: 权重} 字典")
            for a, w in target_weights.items():
                if not np.isfinite(float(w)):
                    raise ValueError(f"target_weights[{a}] 非有限: {w!r}")
        n = max(len(rows), 1)
        weights = target_weights or {name: 1.0 / n for name in rows}
        plan = []
        for name, r in rows.items():
            w = weights.get(name, 1.0 / n)
            target = total * w
            delta = target - r["equity"]
            action = "hold"
            if delta > 0.01:
                action = "transfer_in"
            elif delta < -0.01:
                action = "transfer_out"
            if not r["is_real"]:
                r["engine"].capital = target  # paper/sim 直接生效
            plan.append({"account": name, "is_real": r["is_real"],
                         "current_equity": round(r["equity"], 2),
                         "target_equity": round(target, 2),
                         "delta": round(delta, 2), "action": action})
        out = {"ts": str(_dt.datetime.now()), "total": round(total, 2),
               "plan": plan}
        self._last_rebalance = out
        if write_log:
            try:
                hist = []
                if os.path.exists(write_log):
                    with open(write_log, "r", encoding="utf-8") as _f:
                        hist = json.load(_f)  # noqa
                hist.append(out)
                with open(write_log, "w", encoding="utf-8") as _f:
                    json.dump(hist[-50:], _f, ensure_ascii=False, indent=2,
                              default=str)
            except Exception:
                pass
        return out



def make_multi_engine(config: dict):
    """从配置 dict 构建多账户引擎。

    config = {
      "symbols": [全局标的池],
      "routes": {"by_asset": {"stock":"A","future":"B"},
                 "by_prefix": {"600":"A"}, "default": "A"},
      "default_account": "A",
      "accounts": [
        {"name":"A","broker":"paper","strategy":"sma_cross",
         "symbols":[],"capital":1e6,"live":False,"freq":"daily","risk":{}},
        ...
      ]
    }
    """
    engines = {}
    for a in config.get("accounts", []):
        eng = make_live_engine(
            a.get("broker", "paper"),
            a.get("strategy", "sma_cross"),
            a.get("symbols", []),
            capital=a.get("capital", 1_000_000.0),
            lookback=a.get("lookback", 120),
            live=a.get("live", False),
            freq=a.get("freq", "daily"),
            risk_limits=a.get("risk"),
            sl_pct=a.get("sl_pct"),
            tp_pct=a.get("tp_pct"),
            trailing_pct=a.get("trailing_pct"),
        )
        engines[a["name"]] = eng
    if not engines:
        raise ValueError("make_multi_engine 至少需要一个账户配置")
    router = AccountRouter(config.get("routes"),
                           default=config.get("default_account", list(engines)[0]))
    return MultiLiveEngine(engines, router, symbols=config.get("symbols", []),
                           notify=config.get("notify"))


__all__ = ["AccountRouter", "MultiLiveEngine", "make_multi_engine"]
