"""统一回测调度器：按资产类别自动路由 数据/策略/引擎。

用法：
    from lianghua.backtest.runner import run_backtest
    res = run_backtest("RB2410.SHF", "2023-01-01", "2024-12-31", strategy="breakout")
    res = run_backtest("510050C3000.SH", "2023-01-01", "2024-12-31", strategy="option_call_trend")
    res = run_backtest("110011.OF", "2023-01-01", "2024-12-31", strategy="sma_cross")
"""
from __future__ import annotations

from ..core.assets import AssetType, detect_asset_type
from ..data.gateway import DataGateway
from ..risk.manager import RiskManager
from ..risk.cost import build_cost
from .engine import BacktestEngine, BacktestResult
from .future_engine import FutureEngine
from .option_engine import OptionBacktest
from ..strategy.registry import get_strategy, STRATEGY_REGISTRY, STRATEGY_INFO, STRATEGY_NAMES
from ..option.strategy import get_option_strategy, REGISTRY as OPTION_REGISTRY
from ..fund.backtest import FundBacktest, DCABacktest

# 各资产支持的策略名（用于 UI/CLI 提示）
SUPPORTED = {
    "stock": list(STRATEGY_REGISTRY.keys()),
    "fund": list(STRATEGY_REGISTRY.keys()) + ["dca"],
    "future": list(STRATEGY_REGISTRY.keys()),
    "option": list(OPTION_REGISTRY.keys()),
}


def run_backtest(
    symbol: str,
    start: str,
    end: str,
    strategy: str = "sma_cross",
    asset: str | None = None,
    init_cash: float = 1_000_000.0,
    stop_loss: float = 0.10,
    risk: RiskManager | None = None,
    cost: str | None = None,        # 成本预设: stock/future/option/fund，None=引擎默认
    gw=None,                        # 注入数据网关（离线验证/统一编排用），None=新建默认网关
    **engine_kwargs,
) -> BacktestResult:
    """一键回测任意资产。自动识别类别并选择引擎。"""
    at = AssetType(asset) if asset else detect_asset_type(symbol)
    gw = gw or DataGateway()
    risk = risk or RiskManager(stop_loss=stop_loss)
    # stop_loss / cost 仅用于风控与成本，不转发给引擎构造器
    engine_kwargs.pop("stop_loss", None)
    engine_kwargs.pop("cost", None)
    cost_model = build_cost(cost) if cost else None
    df = gw.fetch(symbol, start, end, asset=at)

    if at == AssetType.OPTION:
        strat = get_option_strategy(strategy)
        signals = strat.generate_signals(df)
        engine = OptionBacktest(init_cash=init_cash, risk=risk, **engine_kwargs)
        return engine.run(df, signals, symbol)

    if at == AssetType.FUTURE:
        strat = get_strategy(strategy)
        signals = strat.generate_signals(df)
        engine = FutureEngine(init_cash=init_cash, risk=risk, cost=cost_model, **engine_kwargs)
        return engine.run(df, signals, symbol)

    if at == AssetType.FUND and strategy == "dca":
        engine = DCABacktest(init_cash=init_cash, risk=risk, **engine_kwargs)
        return engine.run(df, symbol)

    # 股票 / 基金(一次性)
    strat = get_strategy(strategy)
    signals = strat.generate_signals(df)
    engine = BacktestEngine(init_cash=init_cash, risk=risk, cost=cost_model, **engine_kwargs)
    return engine.run(df, signals)
