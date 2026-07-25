"""perf/attribution 单元测试：守卫 + 汇总恒等式 + 分组聚合。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.perf.attribution import (
    attribution_summary,
    brinson_attribution,
    sector_attribution,
)


def _data():
    port_w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    bench_w = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    rp = pd.Series({"A": 0.10, "B": 0.05, "C": -0.02})
    rb = pd.Series({"A": 0.08, "B": 0.06, "C": -0.01})
    return port_w, bench_w, rp, rb


def test_brinson_basic_sum():
    df = brinson_attribution(*_data())
    assert set(df.columns) == {"配置效应", "选择效应", "交叉项", "总效应"}
    assert len(df) == 3
    # 总效应 == 组合收益权重和 - 基准收益权重和
    pw, bw, rp, rb = _data()
    active = (pw * rp).sum() - (bw * rb).sum()
    assert float(df["总效应"].sum()) == pytest.approx(active)


def test_brinson_rejects_empty_common():
    pw = pd.Series({"A": 1.0})
    bw = pd.Series({"B": 1.0})
    rp = pd.Series({"A": 0.1})
    with pytest.raises(ValueError):
        brinson_attribution(pw, bw, rp)


def test_brinson_rejects_nan():
    pw, bw, rp, rb = _data()
    rp = rp.copy()
    rp.iloc[0] = np.nan
    with pytest.raises(ValueError):
        brinson_attribution(pw, bw, rp, rb)


def test_attribution_summary_identity():
    s = attribution_summary(*_data())
    assert s["总效应"] == pytest.approx(s["主动收益"])
    assert s["恒等式残差"] == pytest.approx(0.0, abs=1e-9)


def test_sector_attribution_aggregates():
    pw, bw, rp, rb = _data()
    groups = {"A": "股", "B": "股", "C": "债"}
    out = sector_attribution(pw, bw, rp, groups, rb)
    assert set(out.index) == {"股", "债"}
    # 股分组总效应 == A+B 之和
    df = brinson_attribution(pw, bw, rp, rb)
    assert out.loc["股", "总效应"] == pytest.approx(
        df.loc[["A", "B"], "总效应"].sum())
    assert out.loc["债", "总效应"] == pytest.approx(df.loc["C", "总效应"])


def test_sector_attribution_bad_groups():
    pw, bw, rp, rb = _data()
    with pytest.raises(TypeError):
        sector_attribution(pw, bw, rp, "notadict", rb)
    with pytest.raises(ValueError):
        sector_attribution(pw, bw, rp, {"X": "其它"}, rb)
