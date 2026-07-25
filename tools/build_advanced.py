# -*- coding: utf-8 -*-
"""第十三轮深化：生成 execution/ 下 4 个新/改文件（规避沙箱对 execution/ 不落盘）。"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(r"E:\project\lianghua\lianghua\execution")

# ============ 1. intraday.py ============
intraday = '''"""盘中分钟级信号与监控（第十三轮·深化方向一：盘中分钟级信号）。

- `minute_signal(symbol, strategy, gateway, freq)`：取当日分钟线生成信号。
- `watch(...)`：轻量持续监控回调（不依赖 broker，仅看信号）。
- `IntradayEngine`：分钟级实盘引擎（默认 5min，决策价用实时快照）。
"""
from __future__ import annotations

import datetime as _dt
import time

import numpy as np

from ..core.assets import AssetType, detect_asset_type
from .live import LiveEngine, _resolve_strategy, _asset_val


def minute_signal(symbol, strategy, gateway, freq: str = "5min",
                  lookback: int = 240, asset=None):
    """取当日分钟线，策略生成信号。返回 (signal, last_price, df)。"""
    strat = _resolve_strategy(strategy)
    at = asset or detect_asset_type(symbol)
    today = _dt.date.today().strftime("%Y-%m-%d")
    df = gateway.fetch_minute(symbol, today, asset=_asset_val(at),
                              freq=freq, bars=lookback)
    sig = strat.generate_signals(df)
    last = int(np.sign(float(sig.iloc[-1]))) if len(sig) else 0
    last_price = float(df["close"].iloc[-1]) if len(df) else 0.0
    return last, last_price, df


def watch(symbols, strategy, gateway, freq: str = "5min", lookback: int = 240,
          interval: float = 60.0, on_signal=None, max_cycles=None):
    """轻量持续监控：仅看分钟信号，不依赖 broker，触发 on_signal(symbol, signal, price)。"""
    strat = _resolve_strategy(strategy)
    cycles = 0
    while True:
        for sym in symbols:
            try:
                sg, px, _ = minute_signal(sym, strat, gateway, freq=freq,
                                          lookback=lookback)
                if on_signal:
                    on_signal(sym, sg, px)
            except Exception as e:
                if on_signal:
                    on_signal(sym, 0, 0.0, error=str(e))
        cycles += 1
        if max_cycles is not None and cycles >= int(max_cycles):
            break
        if interval > 0:
            time.sleep(interval)


class IntradayEngine(LiveEngine):
    """分钟级实盘引擎（盘中）。默认 freq=5min，决策价用实时快照 live_quote。"""

    def __init__(self, *args, freq: str = "5min", **kwargs):
        kwargs.pop("freq", None)
        super().__init__(*args, **kwargs, freq=freq)


__all__ = ["minute_signal", "watch", "IntradayEngine"]
'''

# ============ 2. router.py ============
router = '''"""多账户路由（第十三轮·深化方向三：多账户路由）。

把一组标的按规则路由到不同 broker 账户，每个账户可配置独立策略/资金/风控。
典型场景：真实柜台小资金实盘 + paper 大资金演练；股票户 vs 期货户分账。
"""
from __future__ import annotations

import datetime as _dt

from .live import LiveEngine, make_live_engine
from .order_book import OrderBook


class AccountRouter:
    """路由规则：by_asset(资产类型) > by_prefix(代码前缀) > default。"""

    def __init__(self, routes=None, default: str = "default"):
        r = routes or {}
        self.by_asset = r.get("by_asset", {})
        self.by_prefix = r.get("by_prefix", {})
        self.default = r.get("default", default)

    def route(self, symbol, asset_type) -> str:
        atv = asset_type.value if hasattr(asset_type, "value") else str(asset_type)
        if atv in self.by_asset:
            return self.by_asset[atv]
        for pre, acc in self.by_prefix.items():
            if str(symbol).startswith(str(pre)):
                return acc
        return self.default

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
        )
        engines[a["name"]] = eng
    if not engines:
        raise ValueError("make_multi_engine 至少需要一个账户配置")
    router = AccountRouter(config.get("routes"),
                           default=config.get("default_account", list(engines)[0]))
    return MultiLiveEngine(engines, router, symbols=config.get("symbols", []),
                           notify=config.get("notify"))


__all__ = ["AccountRouter", "MultiLiveEngine", "make_multi_engine"]
'''

# ============ 3. schedule.py ============
schedule = '''"""交易时段判断（第十三轮·定时调仓辅助）。"""
from __future__ import annotations

import datetime as _dt

TRADING_AM = (_dt.time(9, 30), _dt.time(11, 30))
TRADING_PM = (_dt.time(13, 0), _dt.time(15, 0))


def is_trading_session(now=None):
    """返回 (bool, str)。周一~周五 09:30-11:30 / 13:00-15:00 为交易中。"""
    now = now or _dt.datetime.now()
    if now.weekday() >= 5:
        return False, "周末休市"
    t = now.time()
    if TRADING_AM[0] <= t <= TRADING_AM[1]:
        return True, "盘中(上午)"
    if TRADING_PM[0] <= t <= TRADING_PM[1]:
        return True, "盘中(下午)"
    if t < TRADING_AM[0]:
        return False, "盘前"
    if TRADING_AM[1] < t < TRADING_PM[0]:
        return False, "午间休市"
    return False, "盘后"


def next_session(now=None):
    """返回下一个交易时段开始时间（datetime）。"""
    now = now or _dt.datetime.now()
    for (start, _) in (TRADING_AM, TRADING_PM):
        dt = now.replace(hour=start.hour, minute=start.minute,
                         second=0, microsecond=0)
        if dt > now:
            return dt
    d = now.date() + _dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    return _dt.datetime.combine(d, TRADING_AM[0])


__all__ = ["is_trading_session", "next_session", "TRADING_AM", "TRADING_PM"]
'''

# ============ 4. live.py (含 freq 分钟级 + run_forever) ============
live = '''"""实盘/纸面交易执行引擎（第十二/十三轮·连接实盘 + 盘中分钟级）。

把「策略信号 -> 盘前风控 -> broker 下单 -> 订单账本 -> 账户快照」串成
单步可重复调用的 LiveEngine：
- freq="daily"        : 日线调仓（默认）。
- freq in {1min,5min} : 盘中分钟级，决策价用实时快照 live_quote。
- paper/sim broker    : 真实行情 + 模拟成交，零风险（默认）。
- qmt/pt broker       : 真实柜台，必须显式 live=True（双保险）。
"""
from __future__ import annotations

import datetime as _dt
import time

import numpy as np

from ..core.assets import AssetType, detect_asset_type
from .brokers.base import Order
from .pre_trade import pre_trade_check
from .order_book import OrderBook


def _asset_val(at):
    return at.value if hasattr(at, "value") else at


def _resolve_strategy(name_or_fn):
    """字符串 -> registry 策略实例；策略对象/可调用对象直接返回。"""
    if not isinstance(name_or_fn, str) and (
            hasattr(name_or_fn, "generate_signals") or callable(name_or_fn)):
        return name_or_fn
    from ..strategy.registry import get_strategy
    return get_strategy(name_or_fn)


class LiveEngine:
    """单引擎多标的实盘/纸面执行器（等权目标仓位，信号驱动调仓）。"""

    LOT = {AssetType.STOCK: 100, AssetType.FUND: 100,
           AssetType.FUTURE: 1, AssetType.OPTION: 1}

    def __init__(self, broker, gateway, strategy, symbols,
                 capital: float = 1_000_000.0, risk_limits: dict | None = None,
                 mode: str = "paper", lookback: int = 120, notify=None,
                 order_db: str = "live_orders.db", live: bool = False,
                 asset_map: dict | None = None, freq: str = "daily"):
        self.broker = broker
        self.gateway = gateway
        self.strategy = _resolve_strategy(strategy)
        self.symbols = list(symbols)
        self.capital = float(capital)
        self.risk_limits = dict(risk_limits or {})
        self.mode = mode
        self.lookback = int(lookback)
        self.notify = notify
        self.book = OrderBook(order_db)
        self.live = bool(live)
        self.asset_map = dict(asset_map or {})
        self.freq = freq
        self.realtime = freq != "daily"
        self.killed = False
        self.last_status: dict | None = None
        self._check_live_safety()

    # ---------- 安全 ----------
    def _check_live_safety(self):
        """真实柜台双保险：QmtBroker/PtBroker 或 mode=live 必须显式 live=True。"""
        is_real = type(self.broker).__name__ in ("QmtBroker", "PtBroker")
        if (self.mode == "live" or is_real) and not self.live:
            raise RuntimeError(
                "真实柜台/实盘模式需要显式 live=True（默认拒绝，防误触真实资金）。"
                "请确认账户、风控与策略后再传 live=True。")

    # ---------- 数据 / 信号 ----------
    def _asset(self, sym: str) -> AssetType:
        a = self.asset_map.get(sym)
        if a is not None:
            return a if isinstance(a, AssetType) else AssetType(a)
        return detect_asset_type(sym)

    def _recent(self, sym: str):
        at = self._asset(sym)
        if self.realtime:
            today = _dt.date.today().strftime("%Y-%m-%d")
            df = self.gateway.fetch_minute(sym, today, asset=_asset_val(at),
                                           freq=self.freq, bars=self.lookback)
            return df, at
        end = _dt.date.today()
        start = end - _dt.timedelta(days=self.lookback * 2)
        df = self.gateway.fetch(sym, str(start), str(end), freq="daily",
                                asset=_asset_val(at))
        return df, at

    def _signal(self, sym: str):
        df, at = self._recent(sym)
        sig = self.strategy.generate_signals(df)
        last = int(np.sign(float(sig.iloc[-1]))) if len(sig) else 0
        last_close = float(df["close"].iloc[-1]) if len(df) else 0.0
        if self.realtime:
            try:
                q = self.gateway.live_quote(sym, asset=_asset_val(at))
                p = float(q.get("price", 0.0) or 0.0)
                if p > 0:
                    last_close = p
            except Exception:
                pass
        return last, at, last_close, df

    # ---------- 仓位 ----------
    def _target_shares(self, at: AssetType, price: float,
                       signal: int, budget: float) -> int:
        if signal == 0 or price <= 0:
            return 0
        if signal < 0 and at not in (AssetType.FUTURE, AssetType.OPTION):
            return 0  # 股票/基金不裸卖空
        lot = self.LOT.get(at, 100)
        raw = int(budget / price / lot) * lot
        return raw if signal > 0 else -raw

    # ---------- 主循环 ----------
    def step(self) -> dict:
        """一次完整调仓：取数 -> 信号 -> 风控 -> 下单 -> 账本 -> 快照。"""
        if self.killed:
            return {"mode": self.mode, "freq": self.freq, "killed": True,
                    "actions": [], "account": {}, "orders": [],
                    "ts": str(_dt.datetime.now())}
        self.broker.connect()
        actions: list[dict] = []
        quotes: dict[str, float] = {}
        signals: dict[str, tuple] = {}
        for sym in self.symbols:
            try:
                sg, at, px, _ = self._signal(sym)
                signals[sym] = (sg, at, px)
                quotes[sym] = px
            except Exception as e:  # 单标的失败不拖垮整体
                actions.append({"symbol": sym, "action": "skip",
                                "reason": "数据/信号失败: %s" % e})
        mv = {}
        pos_now = getattr(self.broker, "positions", {}) or {}
        for s, p in pos_now.items():
            mv[s] = abs(getattr(p, "qty", 0)) * quotes.get(s, getattr(p, "avg_price", 0.0))
        acct = self.broker.get_account(mv)
        n_active = max(sum(1 for v in signals.values() if v[0] != 0), 1)
        budget = float(acct.get("cash", self.capital)) / n_active
        for sym, (sg, at, px) in signals.items():
            cur = acct.get("positions", {}).get(sym, 0)
            tgt = self._target_shares(at, px, sg, budget)
            delta = tgt - cur
            if delta == 0:
                actions.append({"symbol": sym, "action": "hold",
                                "signal": sg, "qty": cur, "price": px})
                continue
            side = "BUY" if delta > 0 else "SELL"
            order = Order(sym, side, abs(int(delta)), px, asset_type=at)
            ok, reason = pre_trade_check(order, acct, self.risk_limits)
            if not ok:
                actions.append({"symbol": sym, "action": "reject",
                                "signal": sg, "side": side,
                                "qty": abs(delta), "price": px, "reason": reason})
                continue
            try:
                res = self.broker.submit(order)
                fp = float(res.get("fill_price", px) or px)
                oid = self.book.record(sym, side, abs(int(delta)), px,
                                       asset=_asset_val(at),
                                       status=str(res.get("status", "filled")),
                                       fill_price=fp,
                                       broker=type(self.broker).__name__,
                                       detail=str(res)[:300])
                actions.append({"symbol": sym, "action": "trade", "signal": sg,
                                "side": side, "qty": abs(delta), "price": px,
                                "fill_price": fp, "order_id": oid})
                if self.notify:
                    try:
                        self.notify("[live] %s %s %d @ %.3f" % (sym, side, abs(delta), px))
                    except Exception:
                        pass
            except Exception as e:
                actions.append({"symbol": sym, "action": "error", "side": side,
                                "qty": abs(delta), "price": px, "reason": str(e)})
        mv2 = {}
        for s, p in (getattr(self.broker, "positions", {}) or {}).items():
            mv2[s] = abs(getattr(p, "qty", 0)) * quotes.get(s, getattr(p, "avg_price", 0.0))
        acct2 = self.broker.get_account(mv2)
        self.last_status = self._status(quotes, acct2, actions)
        return self.last_status

    def _status(self, quotes: dict, acct: dict, actions: list) -> dict:
        return {"mode": self.mode, "freq": self.freq, "killed": self.killed,
                "ts": str(_dt.datetime.now()), "quotes": quotes,
                "account": acct, "actions": actions,
                "orders": self.book.recent(10)}

    def run(self, max_steps: int = 1, interval: float = 0.0) -> list[dict]:
        out = []
        for _ in range(int(max_steps)):
            if self.killed:
                break
            out.append(self.step())
            if interval > 0:
                time.sleep(interval)
        return out

    def run_forever(self, interval: float = 60.0, stop_event=None,
                    max_cycles: int | None = None) -> dict:
        """持续轮询调仓（盘中分钟级常用）。stop_event 为 threading.Event 时可外部停止。"""
        cycles = 0
        while not self.killed:
            if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                break
            self.step()
            cycles += 1
            if max_cycles is not None and cycles >= int(max_cycles):
                break
            if interval > 0:
                time.sleep(interval)
        return self.last_status or {}

    def kill(self) -> dict:
        self.killed = True
        return {"killed": True}

    def status(self) -> dict:
        return self.last_status or {"mode": self.mode, "freq": self.freq,
                                    "killed": self.killed, "actions": [],
                                    "account": {}, "orders": []}

    def __repr__(self):
        return ("LiveEngine(broker=%s, mode=%s, freq=%s, symbols=%s, live=%s)" %
                (type(self.broker).__name__, self.mode, self.freq,
                 self.symbols, self.live))


def make_live_engine(broker_kind: str = "paper", strategy="sma_cross",
                     symbols=("600519.SH",), capital: float = 1_000_000.0,
                     gateway=None, risk_limits: dict | None = None,
                     lookback: int = 120, live: bool = False,
                     freq: str = "daily", **broker_cfg):
    """工厂：一行创建实盘引擎。qmt/pt 需 live=True（真实柜台双保险）。"""
    from .brokers import make_broker
    if gateway is None:
        from ..data.gateway import DataGateway
        gateway = DataGateway()
    is_real = broker_kind in ("qmt", "pt")
    if is_real and not live:
        raise RuntimeError(
            "broker=%s 是真实柜台，必须显式 live=True 才能创建引擎。" % broker_kind)
    kw = dict(broker_cfg)
    if broker_kind in ("sim", "paper"):
        kw.setdefault("init_cash", capital)
    broker = make_broker(broker_kind, **kw)
    return LiveEngine(broker, gateway, strategy, list(symbols), capital=capital,
                      risk_limits=risk_limits,
                      mode=("live" if is_real else "paper"),
                      lookback=lookback, live=(live or is_real), freq=freq)


__all__ = ["LiveEngine", "make_live_engine"]
'''

files = {
    ROOT / "intraday.py": intraday,
    ROOT / "router.py": router,
    ROOT / "schedule.py": schedule,
    ROOT / "live.py": live,
}
for p, txt in files.items():
    p.write_text(txt, encoding="utf-8")
    print("WROTE", p, len(txt))
print("DONE")
