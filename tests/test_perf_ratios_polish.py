"""绩效比率（perf）：负底数幂崩溃修复 + inf 泄漏修复 + 空交集/NaN 守卫 + summary 快照。

续做轮 57（perf 集群）：
- 隐性崩溃：ratios.cagr / annualize.annual_return 在净值跌破/归零时
  total<=0，做分数幂抛 ValueError；现安全降级为 0.0。
- 隐性 inf 泄漏：dist.tail_ratio 左尾为 0 时返回 inf，污染下游指标字典；
  现用 1e-12 兜底返回有限值。
- 隐性 NaN：benchmark 空交集、contrib 列错位、extra._aligned 基准含 NaN，
  原本产出 NaN/inf；现显式守卫或填充。
- 新需求：ratios.summary(equity, benchmark) 一次性安全绩效快照（无 NaN/inf）。
"""
import numpy as np
import pandas as pd
import pytest

from lianghua.perf import ratios, annualize, dist, benchmark, contrib, extra


def _eq(values):
    return pd.Series(values, index=pd.date_range("2023-01-01", periods=len(values), freq="D"))


def test_cagr_no_crash_when_equity_collapses():
    # 净值从 100 跌到 0 -> total=0 -> 不应抛 ValueError
    eq = _eq([100.0, 80.0, 0.0])
    assert ratios.cagr(eq) == 0.0
    assert annualize.annual_return(eq) == 0.0
    # 负底数的分数幂（净值变负）同样安全
    eq2 = _eq([100.0, -50.0])
    assert ratios.cagr(eq2) == 0.0
    assert annualize.annual_return(eq2) == 0.0


def test_annualize_ratio_guards_zero_period():
    assert annualize.annualize_ratio(1.5, 0) == 0.0
    assert annualize.annualize_ratio(1.5, -1) == 0.0
    assert annualize.annualize_ratio(2.0, 12) > 0


def test_dist_tail_ratio_no_inf():
    # 全部正收益 -> 左尾为 0，原先返回 inf；现应为有限值
    r = _eq([100.0, 101.0, 102.0, 103.0, 104.0])
    tr = dist.tail_ratio(r)
    assert np.isfinite(tr)
    assert not np.isinf(tr)


def test_benchmark_empty_overlap_returns_finite():
    a = _eq([100.0, 101.0, 102.0])  # 2023-01-01..03
    b = pd.Series([200.0, 201.0, 202.0],
                  index=pd.date_range("2024-01-01", periods=3, freq="D"))  # 不同区间 -> 交集为空
    assert np.isfinite(benchmark.excess_return(a, b))
    assert np.isfinite(benchmark.tracking_error(a, b))
    assert benchmark.excess_return(a, b) == 0.0


def test_contrib_handles_column_mismatch():
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    # asset_ret 只含 A，缺 B -> 原先 NaN 污染；现 B 填充 0
    ret = pd.DataFrame({"A": [0.01, 0.02]}, index=pd.date_range("2023-01-01", periods=2, freq="D"))
    c = contrib.time_contrib(w, ret)
    assert c.notna().all()
    assert (c == ret["A"] * 0.5).all()
    with pytest.raises(ValueError):
        contrib.time_contrib(pd.Series([1.0], index=["Z"]), ret)  # 完全无交集


def test_extra_aligned_drops_nan_benchmark():
    eq = _eq([100.0, 101.0, 102.0, 103.0])
    bm = pd.Series([100.0, np.nan, 102.0, 103.0],
                   index=pd.date_range("2023-01-01", periods=4, freq="D"))
    r, b = extra._aligned(eq, bm)
    assert r.notna().all() and b.notna().all()
    # Treynor 不应因基准 NaN 产出 NaN
    assert np.isfinite(extra.treynor_ratio(eq, bm))


def test_summary_is_json_safe():
    eq = _eq([100.0, 102.0, 101.0, 104.0, 103.0, 105.0])
    bm = _eq([100.0, 101.0, 100.5, 102.0, 101.5, 103.0])
    s = ratios.summary(eq, bm)
    assert "cagr" in s and "max_drawdown" in s and "up_capture" in s
    for v in s.values():
        assert np.isfinite(v), f"summary 含非有限值: {v}"
