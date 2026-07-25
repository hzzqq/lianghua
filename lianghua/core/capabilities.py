"""Master 能力索引：运行时发现平台全部可调用模块。

UI / CLI / 自驱动循环都从这里拿"现在平台能做什么"，
避免把能力列表硬编码在多处。策略与优化器动态取自注册表，
其余模块按文件静态登记（与迭代构建一致）。
"""
from __future__ import annotations

from ..strategy.registry import STRATEGY_NAMES, STRATEGY_INFO
from ..portfolio.registry import OPTIMIZER_NAMES, OPTIMIZER_INFO
from ..option.strategy import REGISTRY as OPTION_REGISTRY
from ..option.strategy import OPTION_COMBO_REGISTRY


# 非注册表的模块函数清单（与迭代构建保持一致）
_RISK = {
    "var": ["historical_var", "parametric_var", "cvar", "var_report"],
    "drawdown": ["drawdown_series", "max_drawdown_info"],
    "beta": ["beta", "alpha_annual", "capm_residuals"],
    "correlation": ["corr_matrix", "average_correlation", "rolling_corr"],
    "budget": ["equal_risk_contribution", "target_risk_budget"],
    "stress": ["stress_test"],
    "liquidity": ["amihud", "liquidity_cost"],
    "position_sizing": ["kelly_fraction", "fixed_fractional", "vol_target_size"],
    "stop_loss": ["trailing_stop", "stop_triggered"],
    "take_profit": ["take_profit_levels", "scaled_exit"],
    "limit_order": ["backtest_limit"],
    "regime": ["detect_regime", "regime_stats"],
    "cost": ["build_cost"],
    "manager": ["RiskManager"],
    "tail": ["component_var", "cdar", "downside_deviation", "tail_dependence",
             "risk_contribution"],
    "decomp": ["marginal_var", "incremental_var", "diversification_ratio",
               "portfolio_beta", "systematic_var", "idiosyncratic_var",
               "conditional_beta", "risk_parity_deviation", "concentration_index"],
    "extra": ["entropy_risk", "concentration_risk", "correlation_risk",
              "expected_shortfall", "liquidity_adjusted_var", "regime_var",
              "stress_var", "max_loss_prob"],
}
_PERF = {
    "metrics": ["sharpe", "max_drawdown", "annual_return", "summary", "report", "attribution", "win_rate"],
    "sortino": ["sortino", "calmar", "omega_ratio"],
    "dist": ["skew", "kurtosis", "tail_ratio", "value_at_risk"],
    "annualize": ["annual_return", "annual_vol", "annualize_ratio"],
    "monthly": ["monthly_returns", "monthly_table"],
    "benchmark": ["excess_return", "tracking_error", "information_ratio"],
    "trade_stats": ["trade_stats"],
    "rolling": ["rolling_sharpe", "rolling_volatility", "rolling_return", "rolling_max_drawdown", "rolling_report"],
    "rolling_corr": ["rolling_corr_equity"],
    "attribution": ["brinson_attribution"],
    "bootstrap": ["bootstrap_metric"],
    "contrib": ["time_contrib", "cumulative_contrib"],
    "ratios": ["ulcer_index", "martin_ratio", "gain_to_pain", "cagr",
               "up_down_capture", "k_ratio", "omega_ratio", "calmar_ratio",
               "tail_ratio"],
    "extra": ["burke_ratio", "common_sense_ratio", "sterling_ratio", "pain_index",
              "sharpe_penalized", "return_skew", "return_kurtosis",
              "treynor_ratio", "jensen_alpha", "capm_beta"],
    "extra2": ["payoff_ratio", "hit_ratio", "profit_factor", "outlier_ratio",
               "recovery_factor"],
}
_FACTOR = {
    "engine": ["FactorEngine"],
    "auto_search": ["FactorSearcher"],
    "layer": ["ic_series", "ic_decay", "quantile_returns", "long_short_by_factor", "factor_layer_report"],
    "returns": ["factor_long_short", "factor_ic"],
    "zscore": ["winsorize", "zscore", "neutralize"],
    "combine": ["factor_combine", "factor_autocorr", "factor_neutrality",
                "factor_turnover", "factor_icir", "factor_rank_ic",
                "factor_decay", "factor_winsorize",
                "factor_corr", "factor_orthogonalize", "factor_weight_decay",
                "factor_cross_section", "factor_portfolio", "factor_decay_halflife"],
}
_SIGNAL = {
    "score": ["composite_score"],
    "meta": ["majority_vote", "weighted_meta"],
    "rank": ["rank_signal", "rank_to_position"],
    "scheduler": ["rebalance_dates", "schedule"],
    "engine": ["SignalEngine"],
}
_REPORT = {
    "html_report": ["render_backtest_html"],
    "export_all": ["export_backtest"],
    "tearsheet": ["tearsheet"],
}
_INDICATORS = ["rsi", "macd", "kdj", "boll", "atr", "cci", "obv", "vwap",
               "supertrend", "aroon", "vortex", "trix", "williams_r",
               "cmf", "mfi", "stoch_rsi", "dpo", "ppo",
               "keltner_channels", "donchian_channel", "hull_moving_average",
               "true_strength_index", "chandelier_exit", "zscore",
               "ease_of_movement", "mass_index",
               "adx", "pivot", "heikin_ashi", "renko", "demarker", "klinger",
               "chaikin_volatility", "force_index", "know_sure_thing",
               "zigzag", "price_channels"]
_EXECUTION = ["SimBroker", "PaperBroker", "QmtBroker", "PtBroker", "make_broker",
        "sim_order", "slippage", "risk_check", "notify",
        "bracket_order", "oco_order", "pre_trade_check",
        "LiveEngine", "OrderBook", "make_live_engine",
        "IntradayEngine", "minute_signal", "AccountRouter",
        "MultiLiveEngine", "make_multi_engine", "is_trading_session",
        "WeChatNotifier", "WeComAppNotifier", "NotifyHub", "build_notifier",
        "rebalance"]
_DATA = ["DataGateway", "feature_store", "resample", "universe",
        "list_sources", "fetch_any", "get_universe", "list_universes", "live_quote"]


def list_capabilities() -> dict:
    """返回结构化能力清单。"""
    return {
        "strategies": {n: STRATEGY_INFO.get(n, n) for n in STRATEGY_NAMES},
        "option_strategies": {n: getattr(c, "description", n) for n, c in OPTION_REGISTRY.items()},
        "option_combos": {n: spec["desc"] for n, spec in OPTION_COMBO_REGISTRY.items()},
        "optimizers": {n: OPTIMIZER_INFO.get(n, n) for n in OPTIMIZER_NAMES},
        "risk": _RISK,
        "perf": _PERF,
        "factor": _FACTOR,
        "signal": _SIGNAL,
        "report": _REPORT,
        "indicators": _INDICATORS,
        "execution": _EXECUTION,
        "data": _DATA,
    }


def summary_counts() -> dict:
    """各类别数量统计。"""
    cap = list_capabilities()
    return {
        "strategies": len(cap["strategies"]),
        "option_strategies": len(cap["option_strategies"]),
        "optimizers": len(cap["optimizers"]),
        "risk_functions": sum(len(v) for v in cap["risk"].values()),
        "perf_functions": sum(len(v) for v in cap["perf"].values()),
        "factor_functions": sum(len(v) for v in cap["factor"].values()),
        "signal_functions": sum(len(v) for v in cap["signal"].values()),
        "report_functions": sum(len(v) for v in cap["report"].values()),
        "indicators": len(cap["indicators"]),
    }
