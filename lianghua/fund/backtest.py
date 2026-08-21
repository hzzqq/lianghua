"""基金回测：净值(NAV)驱动，支持一次性申购与定期定额(DCA)。

基金行情的 close 即单位净值。与股票不同，基金通常按"份额"申赎，
T 日申请、T+1 确认，含申购费/赎回费。这里简化为即时按净值成交。
"""
from __future__ import annotations

import math
import pandas as pd

from ..core.assets import AssetType, get_contract_spec
from ..risk.manager import RiskManager
from ..backtest.engine import BacktestResult


class FundBacktest:
    """基金一次性申赎回测（类似股票，但按净值、含申赎费）。"""

    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        buy_fee: float = 0.0015,    # 申购费
        sell_fee: float = 0.0005,   # 赎回费
        risk: RiskManager | None = None,
    ):
        self.init_cash = init_cash
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.risk = risk or RiskManager()
        self.trades: list[dict] = []

    def run(self, df: pd.DataFrame, signals: pd.Series, symbol: str = "") -> BacktestResult:
        get_contract_spec(symbol, AssetType.FUND)  # 校验/预留
        cash = self.init_cash
        shares = 0.0
        equity, dates = [], []
        sig = signals.reindex(df.index).fillna(0)
        last_equity = cash
        for i, row in df.iterrows():
            nav = float(row["close"])
            # 隐性修复：NAV 为 NaN/inf/<=0 时，shares=cost/nav 会污染整个权益曲线为
            # NaN/inf，使 BacktestResult.stats 崩溃或产出静默错误指标。无效净值跳过交易，
            # 沿用上一刻权益（持仓按最近有效净值标记）。
            if math.isfinite(nav) and nav > 0:
                s = int(sig.loc[i])
                if s == 1 and shares == 0 and cash > 0:
                    cost = cash * (1 - self.buy_fee)
                    shares = cost / nav
                    self._log(row["date"], "BUY", nav, shares, cash * self.buy_fee)
                    cash = 0.0
                elif s == -1 and shares > 0:
                    proceed = shares * nav * (1 - self.sell_fee)
                    self._log(row["date"], "SELL", nav, shares, proceed)
                    cash = proceed
                    shares = 0.0
                last_equity = cash + shares * nav
            equity.append(last_equity)
            dates.append(row["date"])
        eq = pd.Series(equity, index=pd.to_datetime(dates), name="equity")
        return BacktestResult(eq, self.trades, sig, df)

    def _log(self, date, side, price, qty, fee_or_cash):
        self.trades.append({
            "date": str(date), "side": side, "nav": round(price, 4),
            "shares": round(qty, 2), "fee_or_cash": round(fee_or_cash, 2),
        })


class DCABacktest:
    """基金定期定额(DCA)回测：每 period 个交易日投入固定金额。"""

    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        amount: float = 10_000.0,    # 每期投入
        period: int = 5,             # 每 N 个交易日投入一次
        buy_fee: float = 0.0015,
        risk: RiskManager | None = None,
    ):
        self.init_cash = init_cash
        self.amount = amount
        self.period = period
        self.buy_fee = buy_fee
        self.risk = risk or RiskManager()
        self.trades: list[dict] = []

    def run(self, df: pd.DataFrame, symbol: str = "") -> BacktestResult:
        get_contract_spec(symbol, AssetType.FUND)
        cash = self.init_cash
        shares = 0.0
        equity, dates = [], []
        signals = pd.Series(0, index=df.index)
        last_equity = cash
        for i, row in df.iterrows():
            nav = float(row["close"])
            # 同上：无效净值(非有限/<=0)跳过本期定投，沿用上一刻权益，避免 NaN/inf 污染。
            if math.isfinite(nav) and nav > 0:
                if i % self.period == 0 and cash >= self.amount:
                    invest = self.amount
                    shares += invest * (1 - self.buy_fee) / nav
                    cash -= invest
                    self._log(row["date"], nav, self.amount)
                last_equity = cash + shares * nav
            equity.append(last_equity)
            dates.append(row["date"])
        eq = pd.Series(equity, index=pd.to_datetime(dates), name="equity")
        return BacktestResult(eq, self.trades, signals, df)

    def _log(self, date, nav, amount):
        self.trades.append({
            "date": str(date), "side": "DCA_BUY", "nav": round(nav, 4),
            "invest": round(amount, 2),
        })
