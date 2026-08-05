"""实盘/纸面交易执行引擎（第十二/十三轮·连接实盘 + 盘中分钟级）。

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


def _usable_price(value) -> float | None:
    """把报价规整成"可用于交易决策的价格"，否则返回 None。

    取数彻底失败时 gateway.live_quote 返回 ``price=NaN``，而 NaN 会安静地穿过所有
    朴素校验：``NaN or 0.0`` 得到的还是 NaN（NaN 为真值）、``NaN <= 0`` 为 False。
    于是 NaN 会一路流进
      - 市值 ``qty * NaN`` -> 整个账户权益变成 NaN；
      - 止损/止盈比较（全部为 False）-> **持仓永远不会被止损**，且没有任何报错。
    这里统一收口：只有有限的正数才算有效报价。
    """
    try:
        px = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(px) or px <= 0:
        return None
    return px


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
                 asset_map: dict | None = None, freq: str = "daily",
                 sl_pct: float | None = None, tp_pct: float | None = None,
                 trailing_pct: float | None = None):
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
        self.sl_pct = sl_pct          # 止损比例，如 0.03 = -3% 平仓
        self.tp_pct = tp_pct          # 止盈比例
        self.trailing_pct = trailing_pct  # 移动止盈（从峰值回撤比例）
        self._peak: dict[str, float] = {}
        # 本轮实时报价的降级原因（symbol -> 原因），由 step 汇总进 status["errors"]
        self._quote_notes: dict[str, str] = {}
        self._last_acct: dict = {}
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

    def _mark_to_market(self, quotes: dict) -> dict:
        """按现有报价折算各持仓市值；报价不可用时退回成本价（绝不产出 NaN）。

        NaN 市值会让 ``get_account`` 算出的整个账户权益变成 NaN——一个数字都不能看，
        而且看不出是哪一只标的出的问题。宁可标在成本价（并由 step 把这件事写进
        ``errors``），也不要让一条坏报价污染全账户。
        """
        mv = {}
        for s, p in (getattr(self.broker, "positions", {}) or {}).items():
            px = (_usable_price(quotes.get(s))
                  or _usable_price(getattr(p, "avg_price", None)) or 0.0)
            mv[s] = abs(getattr(p, "qty", 0)) * px
        return mv

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
            # 实时价拿不到就沿用分钟线最后收盘：这是有意的降级，但不能连原因都吞掉。
            try:
                p = _usable_price(
                    self.gateway.live_quote(sym, asset=_asset_val(at)).get("price"))
            except Exception as e:  # noqa: BLE001
                p = None
                self._quote_notes[sym] = "实时快照异常: %s" % e
            else:
                if p is None:
                    self._quote_notes[sym] = "实时快照无有效价，回退分钟线收盘"
            if p is not None:
                last_close = p
                self._quote_notes.pop(sym, None)
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

    # ---------- 盘中止损/止盈 ----------
    def _check_exits(self, quotes: dict) -> list[dict]:
        """盘中止损/止盈/移动止盈：扫描持仓，触发即平仓并记账。

        支持多头（股票/基金/期货多/期权多）与空头（期货空/期权空）双向。
        返回动作列表，由 step() 合并进 actions。
        """
        if not (self.sl_pct or self.tp_pct or self.trailing_pct):
            return []
        actions: list[dict] = []
        for sym, pos in (getattr(self.broker, "positions", {}) or {}).items():
            qty = int(getattr(pos, "qty", 0))
            if qty == 0:
                continue
            at = getattr(pos, "asset_type", self._asset(sym))
            px = _usable_price(quotes.get(sym))
            if px is None:
                try:
                    px = _usable_price(
                        self.gateway.live_quote(sym, _asset_val(at)).get("price"))
                except Exception as e:  # noqa: BLE001
                    actions.append({"symbol": sym, "action": "exit_skip",
                                    "reason": "报价不可用，本轮退出检查已跳过: %s" % e})
                    continue
            if px is None:
                # 报价缺失/NaN 时绝不能"静默跳过"：调用方会以为止损仍在生效。
                actions.append({"symbol": sym, "action": "exit_skip",
                                "reason": "报价不可用，本轮止损/止盈未执行"})
                continue
            avg = float(getattr(pos, "avg_price", px) or px)
            if self.trailing_pct:
                self._peak[sym] = max(self._peak.get(sym, px), px)
            hit = None
            if qty > 0:  # 多头
                if self.sl_pct and px <= avg * (1 - self.sl_pct):
                    hit = "STOP_LOSS"
                elif self.tp_pct and px >= avg * (1 + self.tp_pct):
                    hit = "TAKE_PROFIT"
                elif self.trailing_pct and px <= self._peak[sym] * (1 - self.trailing_pct):
                    hit = "TRAILING_STOP"
            else:  # 空头（期货/期权）
                if self.sl_pct and px >= avg * (1 + self.sl_pct):
                    hit = "STOP_LOSS"
                elif self.tp_pct and px <= avg * (1 - self.tp_pct):
                    hit = "TAKE_PROFIT"
                elif self.trailing_pct and px >= self._peak[sym] * (1 + self.trailing_pct):
                    hit = "TRAILING_STOP"
            if not hit:
                continue
            side = "SELL" if qty > 0 else "BUY"  # BUY=回补空单
            order = Order(sym, side, abs(qty), px, asset_type=at)
            ok, reason = pre_trade_check(order, self._last_acct, self.risk_limits)
            if not ok:
                actions.append({"symbol": sym, "action": "exit_reject",
                                "hit": hit, "reason": reason})
                continue
            try:
                res = self.broker.submit(order)
                fp = float(res.get("fill_price", px) or px)
                oid = self.book.record(
                    sym, side, abs(qty), px,
                    asset=_asset_val(at),
                    status=str(res.get("status", "filled")), fill_price=fp,
                    broker=type(self.broker).__name__,
                    detail="%s:%s" % (hit, str(res)[:200]))
                actions.append({"symbol": sym, "action": "exit", "hit": hit,
                                "side": side, "qty": abs(qty), "price": px,
                                "fill_price": fp, "order_id": oid})
                if self.notify:
                    try:
                        self.notify("[exit] %s %s %d @ %.3f (%s)" %
                                    (sym, side, abs(qty), px, hit))
                    except Exception:
                        pass
            except Exception as e:
                actions.append({"symbol": sym, "action": "exit_error",
                                "hit": hit, "reason": str(e)})
        return actions

    # ---------- 主循环 ----------
    def step(self) -> dict:
        """一次完整调仓：取数 -> 信号 -> 风控 -> 下单 -> 账本 -> 快照。

        健壮性（实盘关键保障）：connect / get_account / 退出检查等外部依赖调用
        均做失败隔离——单点异常不会让整个调仓循环崩溃，而是降级返回带 errors
        的状态快照，便于监控与自动告警；缺失的现金字段（None）回退到初始资金。
        """
        if self.killed:
            return {"mode": self.mode, "freq": self.freq, "killed": True,
                    "actions": [], "account": {}, "orders": [],
                    "errors": [], "ts": str(_dt.datetime.now())}
        errors: list[str] = []
        actions: list[dict] = []
        quotes: dict[str, float] = {}
        signals: dict[str, tuple] = {}
        self._quote_notes.clear()
        try:
            self.broker.connect()
        except Exception as e:
            return self._status(quotes, {}, actions, ["connect 失败: %s" % e])
        for sym in self.symbols:
            try:
                sg, at, px, _ = self._signal(sym)
                signals[sym] = (sg, at, px)
                quotes[sym] = px
            except Exception as e:  # 单标的失败不拖垮整体
                actions.append({"symbol": sym, "action": "skip",
                                "reason": "数据/信号失败: %s" % e})
        # 盘中模式用实时价决策，实时价拿不到而回落到收盘价必须让调用方看见
        errors.extend("标的 %s 决策价已回退（%s）" % (s, why)
                      for s, why in sorted(self._quote_notes.items()))
        # 补全持仓标的的实时价（退出检查需要）
        for s in (getattr(self.broker, "positions", {}) or {}):
            if s in quotes:
                continue
            try:
                px = _usable_price(
                    self.gateway.live_quote(s, _asset_val(self._asset(s))).get("price"))
            except Exception as e:  # noqa: BLE001
                px, note = None, str(e)
            else:
                note = "报价为空/NaN"
            if px is None:
                errors.append("持仓 %s 取价失败(%s)：本轮市值按成本价标记，"
                              "权益仅供参考，且止损/止盈不会触发" % (s, note))
            else:
                quotes[s] = px
        mv = self._mark_to_market(quotes)
        try:
            acct = self.broker.get_account(mv)
        except Exception as e:
            errors.append("get_account 失败: %s" % e)
            return self._status(quotes, {}, actions, errors)
        self._last_acct = acct
        try:
            exit_actions = self._check_exits(quotes)
        except Exception as e:
            errors.append("退出检查异常: %s" % e)
            exit_actions = []
        n_active = max(sum(1 for v in signals.values() if v[0] != 0), 1)
        cash = acct.get("cash", self.capital)
        if cash is None:
            cash = self.capital  # 柜台未返回现金字段时回退初始资金
        budget = float(cash) / n_active
        actions.extend(exit_actions)
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
        mv2 = self._mark_to_market(quotes)
        try:
            acct2 = self.broker.get_account(mv2)
        except Exception as e:
            errors.append("get_account(复盘) 失败: %s" % e)
            acct2 = acct
        self.last_status = self._status(quotes, acct2, actions, errors)
        return self.last_status

    def _status(self, quotes: dict, acct: dict, actions: list,
                errors: list | None = None) -> dict:
        return {"mode": self.mode, "freq": self.freq, "killed": self.killed,
                "ts": str(_dt.datetime.now()), "quotes": quotes,
                "account": acct, "actions": actions,
                "orders": self.book.recent(10),
                "errors": list(errors or [])}

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
                     freq: str = "daily", sl_pct: float | None = None,
                     tp_pct: float | None = None, trailing_pct: float | None = None,
                     **broker_cfg):
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
                      lookback=lookback, live=(live or is_real), freq=freq,
                      sl_pct=sl_pct, tp_pct=tp_pct, trailing_pct=trailing_pct)


__all__ = ["LiveEngine", "make_live_engine"]
