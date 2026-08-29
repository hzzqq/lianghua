"""自研轻量回测引擎（纯 Pandas，零重依赖）。

设计参考 qtrader 的事件流：信号 -> 组合/风控审核 -> 撮合 -> 记账。
默认在信号日收盘价附近撮合（含滑点），支持佣金 + 印花税 + 风控拦截。
如需与 Backtrader 复用，见 backtrader_adapter.py。

v2 改进：
- 成本统一走 _quote()，修复启用 CostModel 时滑点被重复计入（fill 已含滑点，
  cost_for 又计一次）的隐性错误；无 CostModel 时回退到引擎简化成本。
- _buy 资金约束保证 cost <= cash，且不再记录零股交易。
- BacktestResult 新增 stats()：总收益、交易次数、最大回撤等可观测指标。
- to_frame / stats 对缺列、空序列更稳健。

v3 改进（前视偏差 / look-ahead bias 治理）：
- 信号默认**延迟一期执行**（execution_lag=1）。此前信号与成交同处一根 K 线：
  策略用 t 日收盘价算出信号，引擎又用 t 日收盘价成交——实盘中 t 收盘价只有
  收盘那一刻才知道，无法据此成交，回测收益因此虚高。
- 成交价默认取**次日开盘价**（fill_price="open"，缺 open 列时自动回退收盘价），
  进一步贴近实盘；估值（equity）仍按收盘价计算。
- 保留 execution_lag=0 供研究对照，但会在 stats() 中标 look_ahead=True 警示。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..perf.metrics import max_drawdown
from ..risk.cost import CostModel
from ..risk.manager import RiskManager

_LOT = 100


class BacktestResult:
    """回测结果容器。"""

    def __init__(self, equity: pd.Series, trades: list[dict], signals: pd.Series, df: pd.DataFrame,
                 skipped: int = 0, exec_signals: pd.Series | None = None,
                 execution_lag: int = 1, fill_price: str = "open"):
        self.equity = equity
        self.trades = trades
        self.signals = signals
        self.df = df
        self.skipped = skipped
        # 实际用于下单的信号（已延迟 execution_lag 期），与原始 signals 分开放，便于审计
        self.exec_signals = exec_signals if exec_signals is not None else signals
        self.execution_lag = execution_lag
        self.fill_price = fill_price

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame(index=self.equity.index)
        if self.df is not None and "date" in self.df.columns:
            out["date"] = self.df["date"].values[: len(self.equity)]
        out["equity"] = self.equity.values
        if self.df is not None and "close" in self.df.columns:
            out["close"] = self.df["close"].values[: len(self.equity)]
        return out

    def returns(self) -> pd.Series:
        """返回逐日收益序列（新增能力，供下游绩效复用）。"""
        eq = self.equity
        if eq is None or len(eq) < 2:
            return pd.Series([], dtype=float)
        return eq.pct_change().fillna(0.0)

    def stats(self) -> dict:
        """汇总回测绩效指标，便于审计与对比。"""
        eq = self.equity
        if eq is None or len(eq) == 0:
            return {"total_return": 0.0, "n_trades": len(self.trades),
                    "max_drawdown": 0.0, "final_equity": 0.0}
        start = float(eq.iloc[0])
        end = float(eq.iloc[-1])
        # 隐性修复：首值为 NaN 时 total_return 应为 0 而非 NaN
        total_return = (end - start) / start if (pd.notna(start) and start != 0) else 0.0
        # 隐性修复：含 NaN/inf 的权益会让 max_drawdown 抛错，降级为 0
        md = max_drawdown(eq) if np.isfinite(eq.to_numpy()).all() else 0.0
        return {
            "total_return": total_return,
            "n_trades": len(self.trades),
            "max_drawdown": md,
            "final_equity": end,
            # 可审计：结果是否建立在可执行假设上
            "execution_lag": self.execution_lag,
            "fill_price": self.fill_price,
            "look_ahead": self.execution_lag == 0,
        }


class BacktestEngine:
    """事件驱动风格的逐日回测引擎。"""

    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        commission: float = 0.0003,   # 万三
        slippage: float = 0.001,      # 千一
        tax: float = 0.001,           # 千一印花税（卖出）
        risk: RiskManager | None = None,
        cost: "CostModel | None" = None,
        execution_lag: int = 1,       # 信号延迟 N 期执行（1=次日，杜绝前视偏差）
        fill_price: str = "open",     # 成交价基准：open（实盘）| close（研究对照）
    ):
        self.init_cash = init_cash
        self.commission = commission
        self.slippage = slippage
        self.tax = tax
        self.risk = risk or RiskManager()
        self.cost = cost
        # 前视偏差护栏：lag 必须非负整数；fill_price 非法值回退 open
        try:
            self.execution_lag = max(0, int(execution_lag))
        except (TypeError, ValueError):
            self.execution_lag = 1
        self.fill_price = fill_price if fill_price in ("open", "close") else "open"
        self.trades: list[dict] = []

    # ---------- 报价（成交价 + 费用 + 税），杜绝滑点重复计入 ----------
    def _quote(self, price: float, qty: int, side: str):
        cm = self.cost
        if cm:
            fill = cm.fill_price(price, side)
            notional = abs(fill * qty)
            comm = notional * cm.commission_rate + cm.fixed_fee
            imp = notional * cm.impact
            fee = max(comm + imp, cm.min_fee)
            tax = notional * cm.stamp_tax if side in ("SELL", "SHORT", "COVER") else 0.0
        else:
            if side == "BUY":
                fill = price * (1 + self.slippage)
                fee = abs(fill * qty) * self.commission
                tax = 0.0
            else:
                fill = price * (1 - self.slippage)
                fee = abs(fill * qty) * self.commission
                tax = abs(fill * qty) * self.tax
        return fill, fee, tax

    # ---------- 成交价（实盘用开盘，缺 open 列时回退收盘） ----------
    def _trade_price(self, row, close_px: float) -> float:
        if self.fill_price == "open":
            try:
                o = float(row["open"]) if "open" in row.index else float("nan")
            except (TypeError, ValueError):
                o = float("nan")
            if np.isfinite(o) and o > 0:
                return o
        return close_px

    # ---------- 撮合 ----------
    def _buy(self, price: float, qty: int, cash: float, date):
        fill, fee, tax = self._quote(price, qty, "BUY")
        cost = fill * qty + fee + tax
        # 资金不足时向下取整到整百股，确保 cost <= cash
        if cost > cash:
            if self.cost:
                notional_unit = abs(fill * (1 + self.cost.commission_rate + self.cost.impact))
                floor_fee = self.cost.min_fee
            else:
                notional_unit = fill * (1 + self.commission)
                floor_fee = 0.0
            max_q = int((cash - floor_fee) / notional_unit // _LOT) * _LOT
            qty = max(0, min(qty, max_q))
            if qty <= 0:
                return cash, 0, None
            fill, fee, tax = self._quote(price, qty, "BUY")
            cost = fill * qty + fee + tax
        cash -= cost
        trade = {"date": str(date), "side": "BUY", "price": round(fill, 3),
                 "qty": qty, "cash_after": round(cash, 2), "fee": round(fee + tax, 2)}
        return cash, qty, trade

    def _sell(self, price: float, qty: int, cash: float, date, reason: str = "signal"):
        fill, fee, tax = self._quote(price, qty, "SELL")
        proceed = fill * qty - fee - tax
        cash += proceed
        trade = {"date": str(date), "side": "SELL", "price": round(fill, 3),
                 "qty": qty, "reason": reason, "cash_after": round(cash, 2),
                 "fee": round(fee + tax, 2)}
        return cash, 0, trade

    # ---------- 主循环 ----------
    def run(self, df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df 必须是含 close 列的 DataFrame")
        if "close" not in df.columns:
            raise ValueError("df 必须包含 'close' 列")
        has_date = "date" in df.columns
        cash = self.init_cash
        shares = 0
        entry_price = 0.0
        equity, dates = [], []
        last_price = None
        skipped = 0
        pos = 0

        if signals is not None:
            sig = signals.reindex(df.index)
            # 隐性对齐：df 通常只在 `date` 列存日期、index 为默认整数；
            # 若按 index 对齐全为 NaN（信号以日期索引），改用 date 列对齐，避免静默全零。
            if sig.isna().all() and has_date:
                sig = signals.reindex(pd.to_datetime(df["date"]))
                sig.index = df.index
            sig = sig.fillna(0)
        else:
            sig = pd.Series(0, index=df.index)
        # 前视偏差治理：信号延迟 execution_lag 期后再执行。
        # lag=0 表示「当日收盘出信号、当日收盘价成交」——实盘不可执行，仅供研究对照。
        exec_sig = sig.shift(self.execution_lag).fillna(0) if self.execution_lag else sig
        for i, row in df.iterrows():
            dt = row["date"] if has_date else df.index[pos]
            try:
                price = float(row["close"])
            except (TypeError, ValueError):
                price = float("nan")
            # 隐性修复：NaN / 非正价格会让 int(budget/(price*...)) 抛 ValueError、
            # price=0 触发 ZeroDivision；改为跳过当日交易并沿用上一有效价估值。
            if not np.isfinite(price) or price <= 0:
                skipped += 1
                val_price = last_price if last_price is not None else 0.0
                equity.append(cash + shares * val_price)
                dates.append(dt)
                pos += 1
                continue
            last_price = price
            s = int(exec_sig.iloc[pos])
            # 成交价：信号已于 execution_lag 期前产生，用当前 bar 的开盘价成交（实盘可执行）
            tpx = self._trade_price(row, price)

            # 持仓中断损（按收盘价判定，按成交价卖出）
            if shares > 0 and self.risk.check_exit(entry_price, price):
                cash, shares, trade = self._sell(tpx, shares, cash, dt, "stop_loss")
                self._log(trade)
                entry_price = 0.0

            # 买入信号
            if s == 1 and shares == 0:
                size = self.risk.position_size(cash, tpx)
                budget = cash * size
                qty = int(budget / (tpx * (1 + self.slippage + self.commission)) // _LOT) * _LOT
                if qty > 0 and self.risk.check_entry(str(row.get("symbol", "")), cash):
                    cash, shares, trade = self._buy(tpx, qty, cash, dt)
                    if trade is not None:
                        self._log(trade)
                        entry_price = tpx

            # 卖出信号
            elif s == -1 and shares > 0:
                cash, shares, trade = self._sell(tpx, shares, cash, dt, "signal")
                self._log(trade)
                entry_price = 0.0

            equity.append(cash + shares * price)
            dates.append(dt)
            pos += 1

        eq = pd.Series(equity, index=pd.to_datetime(dates), name="equity")
        return BacktestResult(eq, self.trades, sig, df, skipped=skipped,
                              exec_signals=exec_sig, execution_lag=self.execution_lag,
                              fill_price=self.fill_price)

    def _log(self, trade: dict):
        self.trades.append(trade)
