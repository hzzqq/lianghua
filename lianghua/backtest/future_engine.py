"""期货回测引擎：保证金 / 杠杆 / 多空双向 / 展期。

与股票引擎解耦：期货按合约(手)交易，逐日盯市(mark-to-market)，
占用保证金、支持多空，并在临近交割时自动展期(roll)。
信号约定：1=做多, -1=做空, 0=平仓/观望。
成本：若传入 cost(CostModel) 则按成本模型计算滑点/佣金，否则用自有 slippage/commission_per_hand。
"""
from __future__ import annotations

import pandas as pd

from ..core.assets import AssetType, get_contract_spec
from ..risk.manager import RiskManager
from ..risk.cost import CostModel
from .engine import BacktestResult


class FutureEngine:
    """期货逐日回测引擎。"""

    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        leverage: float = 3.0,           # 目标杠杆（受保证金率上限约束）
        slippage: float = 0.001,
        risk: RiskManager | None = None,
        roll_days: int = 5,              # 距到期少于该天数则展期
        cost: "CostModel | None" = None,
    ):
        self.init_cash = init_cash
        self.leverage = leverage
        self.slippage = slippage
        self.risk = risk or RiskManager()
        self.roll_days = roll_days
        self.cost = cost
        self.trades: list[dict] = []

    def _margin_per_hand(self, price: float, spec) -> float:
        return price * spec.multiplier * spec.margin_rate

    def _target_contracts(self, cash: float, price: float, spec) -> int:
        margin_budget = cash * min(self.leverage * spec.margin_rate, 1.0)
        per = self._margin_per_hand(price, spec)
        if per <= 0:
            return 0
        return int(margin_budget // per)

    def _fill(self, side: str, price: float) -> float:
        if self.cost:
            return self.cost.fill_price(price, side)
        return price * (1 + (self.slippage if side in ("LONG", "BUY") else -self.slippage))

    def _fee(self, side: str, size: float, price: float, spec) -> float:
        if self.cost:
            return self.cost.cost_for(side, price, size, spec.multiplier)
        return size * spec.commission_per_hand

    def run(self, df: pd.DataFrame, signals: pd.Series, symbol: str = "") -> BacktestResult:
        spec = get_contract_spec(symbol, AssetType.FUTURE)
        cash = self.init_cash
        pos = 0              # 持仓手数（正多/负空）
        prev_price = float(df["close"].iloc[0])
        equity, dates = [], []

        sig = signals.reindex(df.index).fillna(0)
        for i, row in df.iterrows():
            price = float(row["close"])
            s = int(sig.loc[i])

            # 逐日盯市
            if pos != 0:
                cash += pos * (price - prev_price) * spec.multiplier

            # 风控：权益不足以覆盖保证金 -> 强平
            required = abs(pos) * self._margin_per_hand(price, spec)
            if required > cash:
                cash -= self._fee("SELL", abs(pos), price, spec)
                pos = 0
                self._log(row["date"], "FORCE_CLOSE", price, 0, cash)

            # 多空切换
            if s == 1 and pos <= 0:
                if pos < 0:
                    cash -= self._fee("BUY", abs(pos), price, spec)
                    self._log(row["date"], "BUY_COVER", price, pos, cash)
                    pos = 0
                target = self._target_contracts(cash, price, spec)
                if target > 0 and self.risk.check_entry(symbol, cash):
                    fill = self._fill("LONG", price)
                    margin = target * self._margin_per_hand(fill, spec)
                    if margin <= cash:
                        cash -= self._fee("LONG", target, price, spec)
                        pos = target
                        self._log(row["date"], "LONG", fill, pos, cash)
            elif s == -1 and pos >= 0:
                if pos > 0:
                    cash -= self._fee("SELL", pos, price, spec)
                    self._log(row["date"], "SELL_CLOSE", price, pos, cash)
                    pos = 0
                target = self._target_contracts(cash, price, spec)
                if target > 0 and self.risk.check_entry(symbol, cash):
                    fill = self._fill("SHORT", price)
                    margin = target * self._margin_per_hand(fill, spec)
                    if margin <= cash:
                        cash -= self._fee("SHORT", target, price, spec)
                        pos = -target
                        self._log(row["date"], "SHORT", fill, pos, cash)

            equity.append(cash)
            dates.append(row["date"])
            prev_price = price

        eq = pd.Series(equity, index=pd.to_datetime(dates), name="equity")
        return BacktestResult(eq, self.trades, sig, df)

    def _log(self, date, side, price, qty, cash):
        self.trades.append({
            "date": str(date), "side": side,
            "price": round(float(price), 3), "qty": int(qty),
            "cash_after": round(float(cash), 2),
        })
