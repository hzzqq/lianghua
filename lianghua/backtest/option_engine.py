"""期权回测引擎：权利金结算 / 多空期权 / 希腊字母可用。

信号约定：1=买入期权(做多波动率/方向), -1=平仓或卖出, 0=持有。
按"手"交易，每手价值 = option_price * 合约乘数(multiplier)。

稳健性设计：
- 佣金按权利金市值比例计（避免低价期权手数爆炸导致佣金失控）
- 权利金 <= PRICE_FLOOR 时不开仓（临近归零的虚值期权）
- 单笔手数设上限，仓位规模以初始资金为基准（避免复利式失控）
"""
from __future__ import annotations

import pandas as pd

from ..core.assets import AssetType, get_contract_spec
from ..risk.manager import RiskManager
from .engine import BacktestResult

PRICE_FLOOR = 1e-3
MAX_LOTS = 5000


class OptionBacktest:
    """期权回测引擎。"""

    def __init__(
        self,
        init_cash: float = 1_000_000.0,
        max_premium_frac: float = 0.2,  # 单笔权利金占初始资金上限
        commission_rate: float = 0.0005,
        slippage: float = 0.001,
        risk: RiskManager | None = None,
    ):
        self.init_cash = init_cash
        self.max_premium_frac = max_premium_frac
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.risk = risk or RiskManager()
        self.trades: list[dict] = []

    def _open_lots(self, cash: float, price: float, mult: float) -> int:
        if price <= PRICE_FLOOR:
            return 0
        budget = self.init_cash * self.max_premium_frac
        lots = int(budget // (price * mult * (1 + self.slippage)))
        return max(0, min(lots, MAX_LOTS))

    def _comm(self, notional: float) -> float:
        return notional * self.commission_rate

    def run(self, df: pd.DataFrame, signals: pd.Series, symbol: str = "") -> BacktestResult:
        spec = get_contract_spec(symbol, AssetType.OPTION)
        mult = spec.multiplier
        cash = self.init_cash
        pos = 0           # 手数（正=多, 负=空）
        equity, dates = [], []

        sig = signals.reindex(df.index).fillna(0)
        for i, row in df.iterrows():
            price = float(row["option_price"])
            s = int(sig.loc[i])

            if s == 1 and pos <= 0:
                if pos < 0:
                    proceed = abs(pos) * price * mult * (1 - self.slippage) - self._comm(abs(pos) * price * mult)
                    cash += proceed
                    self._log(row["date"], "SELL_CLOSE", price, pos, cash)
                    pos = 0
                lots = self._open_lots(cash, price, mult)
                if lots > 0 and self.risk.check_entry(symbol, cash):
                    notional = lots * price * mult * (1 + self.slippage)
                    cash -= notional + self._comm(notional)
                    pos = lots
                    self._log(row["date"], "BUY", price, pos, cash)
            elif s == -1 and pos >= 0:
                if pos > 0:
                    proceed = pos * price * mult * (1 - self.slippage) - self._comm(pos * price * mult)
                    cash += proceed
                    self._log(row["date"], "SELL", price, pos, cash)
                    pos = 0
                else:
                    lots = self._open_lots(cash, price, mult)
                    if lots > 0 and self.risk.check_entry(symbol, cash):
                        notional = lots * price * mult * (1 - self.slippage)
                        cash += notional - self._comm(notional)
                        pos = -lots
                        self._log(row["date"], "SELL_SHORT", price, pos, cash)

            mtm = pos * price * mult
            equity.append(cash + mtm)
            dates.append(row["date"])

        eq = pd.Series(equity, index=pd.to_datetime(dates), name="equity")
        return BacktestResult(eq, self.trades, sig, df)

    def _log(self, date, side, price, qty, cash):
        self.trades.append({
            "date": str(date), "side": side,
            "price": round(float(price), 4), "qty": int(qty),
            "cash_after": round(float(cash), 2),
        })
