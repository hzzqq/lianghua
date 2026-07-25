"""Lianghua Quant —— 轻量模块化量化交易框架。

模仿 vnpy / Backtrader / qtrader 的设计，提供可插拔的
数据 → 策略 → 回测 → 风控 → 绩效 → UI 全链路。
"""
from __future__ import annotations

from .core.constants import VERSION as VERSION
from .core.capabilities import list_capabilities, summary_counts

__version__ = "1.6.0-live"
__author__ = "Lianghua Quant Team"

# 平台能力清单（迭代 1–130 累计）
CAPABILITIES = {
    "asset": ["stock", "fund", "future", "option"],
    "engines": ["engine", "future_engine", "option_engine", "fund_backtest",
                "vectorized", "basket", "orchestrator", "multi_strategy",
                "future_spread"],
    "risk": ["manager", "cost", "var", "beta", "drawdown",
             "correlation", "budget", "stress", "liquidity",
             "position_sizing", "stop_loss", "take_profit", "limit_order",
             "regime"],
    "perf": ["metrics", "rolling", "sortino", "dist", "annualize",
             "attribution", "trade_stats", "monthly", "benchmark",
             "bootstrap", "rolling_corr", "contrib", "tearsheet"],
    "portfolio": ["optimizer", "rebalance", "allweather", "min_variance",
                 "max_diversification", "hrp", "black_litterman",
                 "target_vol", "cppi", "risk_parity_ewm", "clustering"],
    "strategy": ["examples", "turtle", "rsi_strategy", "bollinger", "grid",
                 "mean_reversion_z", "rotation", "vol_target",
                 "dual_momentum", "pair", "kalman", "donchian",
                 "keltner", "coppock", "seasonal", "vol_breakout",
                 "rsi_divergence", "parallel_sar", "adaptive",
                 "ou_mean_reversion", "ml_signal", "ensemble"],
    "factor": ["engine", "auto_search", "layer", "zscore", "returns"],
    "signal": ["engine", "score", "scheduler", "meta", "rank"],
    "data": ["gateway", "resample", "feature_store", "universe"],
    "execution": ["brokers", "notify", "sim_order", "slippage", "risk_check"],
    "report": ["html_report", "export_all", "tearsheet"],
    "ui": ["app", "widgets"],
    "indicators": ["tech", "tech2"],
}

# 第七轮扩展（迭代 101–130）：注册表自动发现，此处仅登记新增大类成员
CAPABILITIES_ITER101_130 = {
    "indicators_tech2": ["supertrend", "aroon", "vortex", "trix", "williams_r",
                          "cmf", "mfi", "stoch_rsi", "dpo", "ppo"],
    "strategy_new": ["supertrend", "aroon", "vortex", "trix", "williams_r",
                     "elder_ray", "ichimoku", "macd_hist", "chaikin", "stoch_rsi"],
    "optimizer_new": ["max_sharpe", "min_cvar", "shrinkage_min_var",
                      "max_entropy", "momentum_score"],
    "perf_ratios": ["ulcer_index", "martin_ratio", "gain_to_pain", "cagr",
                    "up_down_capture", "k_ratio"],
    "risk_tail": ["downside_deviation", "component_var", "cdar"],
    "factor_combine": ["factor_combine", "factor_autocorr"],
}

# 第八轮扩展（迭代 131–140）：注册表自动发现，此处仅登记新增大类成员
CAPABILITIES_ITER131_140 = {
    "strategy_new": ["parabolic_sar", "adx_trend", "cci_signal", "roc",
                     "ultimate_oscillator"],
    "optimizer_new": ["min_tail_risk", "vol_target_opt"],
    "perf_ratios_new": ["omega_ratio", "calmar_ratio", "tail_ratio"],
    "risk_tail_new": ["tail_dependence", "risk_contribution"],
    "factor_combine_new": ["factor_neutrality", "factor_turnover"],
    "selfdrive": ["core/selfdrive.next_gaps", "core/selfdrive.deficit"],
}

# 第九轮扩展（迭代 141–150）：注册表自动发现，此处仅登记新增大类成员
CAPABILITIES_ITER141_150 = {
    "strategy_new": ["pivot_points", "heikin_ashi", "renko_trend", "demark", "klinger"],
    "optimizer_new": ["bayesian_shrinkage", "max_return",
                      "min_track_error", "robust_cov"],
    "factor_combine_new": ["factor_icir"],
}

# 第十轮扩展（迭代 151–180）：注册表自动发现，此处仅登记新增大类成员
CAPABILITIES_ITER151_180 = {
    "indicators_tech3": ["keltner_channels", "donchian_channel", "hull_moving_average",
                          "true_strength_index", "chandelier_exit", "zscore",
                          "ease_of_movement", "mass_index"],
    "perf_extra": ["burke_ratio", "common_sense_ratio", "sterling_ratio", "pain_index",
                   "sharpe_penalized", "return_skew", "return_kurtosis",
                   "treynor_ratio", "jensen_alpha", "capm_beta"],
    "risk_decomp": ["marginal_var", "incremental_var", "diversification_ratio",
                    "portfolio_beta", "systematic_var", "idiosyncratic_var",
                    "conditional_beta", "risk_parity_deviation", "concentration_index"],
    "factor_combine_new2": ["factor_rank_ic", "factor_decay", "factor_winsorize"],
}

# 第十一轮扩展（迭代 181–200，超自驱动目标深化）：期权组合 / 数据多源 / 执行增强 /
# 指标 tech4 / 绩效 extra2 / 风险 extra / 因子 combine 扩展（均由注册表自动发现）
CAPABILITIES_ITER181_200 = {
    "option_combos": ["covered_call", "protective_put", "collar", "straddle",
                      "strangle", "iron_butterfly", "calendar_spread",
                      "ratio_spread", "diagonal_spread", "long_call", "long_put",
                      "vertical_spread", "iron_condor", "butterfly"],
    "data_sources": ["synthetic", "akshare", "baostock", "fetch_any", "get_universe"],
    "execution_extra": ["PaperBroker", "bracket_order", "oco_order", "pre_trade_check"],
    "indicators_tech4": ["adx", "pivot", "heikin_ashi", "renko", "demarker", "klinger",
                         "chaikin_volatility", "force_index", "know_sure_thing",
                         "zigzag", "price_channels"],
    "perf_extra2": ["payoff_ratio", "hit_ratio", "profit_factor", "outlier_ratio",
                    "recovery_factor"],
    "risk_extra": ["entropy_risk", "concentration_risk", "correlation_risk",
                   "expected_shortfall", "liquidity_adjusted_var", "regime_var",
                   "stress_var", "max_loss_prob"],
    "factor_combine_new3": ["factor_corr", "factor_orthogonalize", "factor_weight_decay",
                            "factor_cross_section", "factor_portfolio",
                            "factor_decay_halflife"],
}
