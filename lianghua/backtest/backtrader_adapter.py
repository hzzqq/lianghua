"""可选 Backtrader 适配层。

核心引擎是自研 Pandas 引擎（见 engine.py），此文件为可选适配：
当用户希望复用 Backtrader 生态（122 种指标、Analyzer、优化器）时使用。
需先安装：pip install backtrader
"""
from __future__ import annotations

try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except Exception:
    BACKTRADER_AVAILABLE = False


class BacktraderEngine:
    """基于 Backtrader 的回测引擎（可选）。"""

    def __init__(self, init_cash: float = 1_000_000.0, commission: float = 0.0003):
        if not BACKTRADER_AVAILABLE:
            raise RuntimeError("未安装 backtrader，请先执行: pip install backtrader")
        self.init_cash = init_cash
        self.commission = commission

    def run(self, df, strategy_cls, **kwargs):
        """用 Backtrader 跑回测。

        参数:
            df: 含 date/open/high/low/close/volume 的 DataFrame
            strategy_cls: 继承 bt.Strategy 的策略类
        """
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.init_cash)
        cerebro.broker.setcommission(commission=self.commission)

        data = bt.feeds.PandasData(
            dataname=df, datetime="date", open="open", high="high",
            low="low", close="close", volume="volume", openinterest=None,
        )
        cerebro.adddata(data)
        cerebro.addstrategy(strategy_cls, **kwargs)
        results = cerebro.run()
        return cerebro.broker.getvalue(), results
