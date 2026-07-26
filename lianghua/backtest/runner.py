"""统一回测调度器：按资产类别自动路由 数据/策略/引擎。

用法：
    from lianghua.backtest.runner import run_backtest
    res = run_backtest("RB2410.SHF", "2023-01-01", "2024-12-31", strategy="breakout")
    res = run_backtest("510050C3000.SH", "2023-01-01", "2024-12-31", strategy="option_call_trend")
    res = run_backtest("110011.OF", "2023-01-01", "2024-12-31", strategy="sma_cross")
"""
from __future__ import annotations

import pandas as pd

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


def list_supported_strategies(asset: str | None = None):
    """返回资产类别支持的策略名列表（asset=None 时返回全量映射）。"""
    if asset is None:
        return SUPPORTED
    return SUPPORTED.get(asset, [])


def _resolve_asset(symbol: str, asset: str | None) -> AssetType:
    """识别资产类型，识别失败时给出清晰错误而非抛出内部异常。"""
    try:
        return AssetType(asset) if asset else detect_asset_type(symbol)
    except Exception as e:  # 内部检测异常（如未知后缀）
        raise ValueError(f"无法识别资产类型：symbol={symbol!r}, asset={asset!r}（{e}）")


def _validate_strategy(at: AssetType, strategy: str) -> None:
    """隐性修复：原实现先取数再 get_strategy，策略非法时既白取了数又只抛难懂的 KeyError。
    现先校验策略，给出'可选策略'清单。"""
    supported = SUPPORTED.get(at.value, [])
    if strategy not in supported:
        raise ValueError(
            f"资产 {at.value} 不支持策略 {strategy!r}；可选：{supported}"
        )


def validate_backtest_request(symbol: str, strategy: str,
                              asset: str | None = None) -> dict:
    """new_requirement（新增能力）：回测前预检，返回 {asset_type, strategy,
    supported, ok} 供 UI/CLI 在真正取数前预警，避免无效请求。"""
    at = _resolve_asset(symbol, asset)
    supported = SUPPORTED.get(at.value, [])
    return {
        "asset_type": at.value,
        "strategy": strategy,
        "supported": supported,
        "ok": strategy in supported,
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
    at = _resolve_asset(symbol, asset)
    _validate_strategy(at, strategy)           # 先校验策略，避免白取数据
    gw = gw or DataGateway()
    risk = risk or RiskManager(stop_loss=stop_loss)
    # stop_loss / cost 仅用于风控与成本，不转发给引擎构造器
    engine_kwargs.pop("stop_loss", None)
    engine_kwargs.pop("cost", None)
    cost_model = build_cost(cost) if cost else None
    df = gw.fetch(symbol, start, end, asset=at)
    # 隐性修复：网关可能返回 None/空 DataFrame（代码错、日期错、网络失败），
    # 原代码直接交给引擎，引擎内部才崩且报错不清晰。这里在取数后立即显式拦截。
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        raise ValueError(
            f"未获取到回测数据：symbol={symbol!r}, {start}~{end}（检查代码/日期范围/网络）"
        )

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
