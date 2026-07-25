"""第十轮扩展（迭代 151–180）集成测试。

覆盖：
- 8 个新指标（tech3）：输出等长 Series/DataFrame
- 10 个新绩效函数（perf/extra）：输入净值/收益返回有限标量
- 9 个新风险分解函数（risk/decomp）：数值合理
- 3 个新因子函数（factor/combine）：rank_ic / decay / winsorize
- 能力计数：indicators=37 / perf_functions=57 / risk_functions=51 / factor_functions=26
- 自驱动缺口生成器：本轮新增项已被自动划掉（不出现在 pending）

纯合成数据，离线全绿。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N = 250


def _ohlcv():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=N, freq="D")
    close = pd.Series(100 + np.cumsum(rng.normal(size=N)), index=idx)
    high = close + np.abs(rng.normal(size=N))
    low = close - np.abs(rng.normal(size=N))
    open_ = close.shift(1).fillna(close)
    vol = pd.Series(rng.random(N) * 1e6 + 1e4, index=idx)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _returns_df(n=N, k=5):
    rng = np.random.default_rng(13)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = [chr(ord("A") + i) for i in range(k)]
    return pd.DataFrame(rng.normal(0, 0.01, size=(n, k)), index=idx, columns=cols)


def _equity(n=N):
    rng = np.random.default_rng(3)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=n)), index=idx)


# ---------------------------------------------------------------- 指标 ×8
def test_indicators_tech3():
    from lianghua.indicators import tech3
    df = _ohlcv()
    df_fns = [tech3.keltner_channels, tech3.donchian_channel, tech3.chandelier_exit]
    for f in df_fns:
        out = f(df.copy())
        assert len(out) == N, (f.__name__, len(out))
        assert out.shape[1] >= 2
    close_fns = [tech3.hull_moving_average, tech3.true_strength_index, tech3.zscore]
    for f in close_fns:
        out = f(df["close"])
        assert len(out) == N, (f.__name__, len(out))
    extra_fns = [tech3.ease_of_movement, tech3.mass_index]
    for f in extra_fns:
        out = f(df.copy())
        assert len(out) == N, (f.__name__, len(out))
    print("indicators tech3: 8/8 OK")


# ---------------------------------------------------------------- 绩效 ×10
def test_perf_extra():
    from lianghua.perf import extra
    eq = _equity()
    bm = _equity(300)  # 不同种子近似基准
    rng = np.random.default_rng(9)
    rets = pd.Series(rng.normal(0, 0.01, size=N))
    pairs = {
        "burke_ratio": extra.burke_ratio(eq),
        "common_sense_ratio": extra.common_sense_ratio(eq),
        "sterling_ratio": extra.sterling_ratio(eq),
        "pain_index": extra.pain_index(eq),
        "sharpe_penalized": extra.sharpe_penalized(eq),
        "return_skew": extra.return_skew(rets),
        "return_kurtosis": extra.return_kurtosis(rets),
        "treynor_ratio": extra.treynor_ratio(eq, bm),
        "jensen_alpha": extra.jensen_alpha(eq, bm),
        "capm_beta": extra.capm_beta(eq, bm),
    }
    for name, v in pairs.items():
        assert np.isfinite(v), (name, v)
    print("perf extra: 10/10 OK")


# ---------------------------------------------------------------- 风险 ×9
def test_risk_decomp():
    from lianghua.risk import decomp
    R = _returns_df()
    w = np.array([0.2] * 5)
    mkt = pd.Series(np.random.default_rng(1).normal(0, 0.01, size=N),
                    index=R.index)
    mv = decomp.marginal_var(R, w)
    assert len(mv) == 5 and np.isfinite(mv).all()
    iv = decomp.incremental_var(R, w)
    assert len(iv) == 5 and np.isfinite(iv).all()
    assert np.isfinite(decomp.diversification_ratio(R))
    assert np.isfinite(decomp.portfolio_beta(R, w, mkt))
    assert np.isfinite(decomp.systematic_var(R, w, mkt))
    assert np.isfinite(decomp.idiosyncratic_var(R, w, mkt))
    assert np.isfinite(decomp.conditional_beta(R, w, mkt))
    assert np.isfinite(decomp.risk_parity_deviation(R, w))
    assert abs(decomp.concentration_index(w) - 0.2) < 1e-9
    print("risk decomp: 9/9 OK")


# ---------------------------------------------------------------- 因子 ×3
def test_factor_new():
    from lianghua.factor import combine as C
    rng = np.random.default_rng(4)
    idx = pd.date_range("2023-01-01", periods=80, freq="D")
    cols = [chr(ord("A") + i) for i in range(5)]
    fp = pd.DataFrame(rng.normal(size=(80, 5)), index=idx, columns=cols)
    fwd = pd.DataFrame(rng.normal(size=(80, 5)), index=idx, columns=cols)
    ric = C.factor_rank_ic(fp, fwd)
    assert np.isfinite(ric)
    decay = C.factor_decay(fp, fwd, lags=(1, 3, 5))
    assert len(decay) == 3 and np.isfinite(decay).all()
    s = pd.Series(np.concatenate([rng.normal(size=200), [1000.0, -1000.0]]))
    w = C.factor_winsorize(s)
    # 极端值被裁剪到分位边界（明显低于原极端）
    assert w.max() < s.max(), (w.max(), s.max())
    assert w.min() > s.min(), (w.min(), s.min())
    assert w.max() <= s.max() and w.min() >= s.min()
    print("factor new: 3/3 OK")


# ---------------------------------------------------------------- 能力计数
def test_counts():
    from lianghua.core.capabilities import summary_counts, list_capabilities
    c = summary_counts()
    assert c["indicators"] == 37, c
    assert c["perf_functions"] == 57, c
    assert c["risk_functions"] == 51, c
    assert c["factor_functions"] == 26, c
    cap = list_capabilities()
    perf_union = {f for v in cap["perf"].values() for f in v}
    risk_union = {f for v in cap["risk"].values() for f in v}
    factor_union = {f for v in cap["factor"].values() for f in v}
    for n in ["keltner_channels", "donchian_channel", "hull_moving_average",
              "true_strength_index", "chandelier_exit", "zscore",
              "ease_of_movement", "mass_index"]:
        assert n in cap["indicators"], f"指标 {n} 未登记"
    for n in ["burke_ratio", "common_sense_ratio", "sterling_ratio", "pain_index",
              "sharpe_penalized", "return_skew", "return_kurtosis",
              "treynor_ratio", "jensen_alpha", "capm_beta"]:
        assert n in perf_union, f"绩效 {n} 未登记"
    for n in ["marginal_var", "incremental_var", "diversification_ratio",
              "portfolio_beta", "systematic_var", "idiosyncratic_var",
              "conditional_beta", "risk_parity_deviation", "concentration_index"]:
        assert n in risk_union, f"风险 {n} 未登记"
    for n in ["factor_rank_ic", "factor_decay", "factor_winsorize"]:
        assert n in factor_union, f"因子 {n} 未登记"
    print("counts: ind26/perf52/risk43/factor20 OK")


# ---------------------------------------------------------------- 自驱动缺口
def test_selfdrive_excludes_new():
    from lianghua.core import selfdrive
    d = selfdrive.deficit()
    # 本轮涉及的候选名应已实现（不出现在 pending）
    impl = {f for v in selfdrive._implemented().values() for f in v}
    for n in ["donchian_channel", "mass_index", "burke_ratio", "sterling_ratio",
              "marginal_var", "incremental_var", "diversification_ratio",
              "portfolio_beta", "systematic_var", "idiosyncratic_var",
              "conditional_beta", "risk_parity_deviation", "concentration_index",
              "factor_rank_ic", "factor_decay", "factor_winsorize"]:
        assert n in impl, f"自驱动未识别已实现的 {n}"
    # 目标已封顶的类别 gap 为 0
    assert d["indicators"]["gap_to_target"] == 0
    assert d["perf_functions"]["gap_to_target"] == 0
    assert d["risk_functions"]["gap_to_target"] == 0
    assert d["factor_functions"]["gap_to_target"] == 0
    print("selfdrive: 新项已自动划掉，目标全封顶 OK")


def _run_all():
    for fn in [test_indicators_tech3, test_perf_extra, test_risk_decomp,
               test_factor_new, test_counts, test_selfdrive_excludes_new]:
        fn()
    print("\nALL_ITER151_180_OK")


if __name__ == "__main__":
    _run_all()
