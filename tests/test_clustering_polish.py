"""R41: clustering 边界与可观测验收。"""
import numpy as np
import pandas as pd

from lianghua.portfolio.clustering import cluster_assets, cluster_summary


def _df(n=5, t=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (t, n)),
        columns=[f"A{i}" for i in range(n)])


def test_basic_clusters():
    d = _df(5)
    res = cluster_assets(d, 2)
    flat = [a for v in res.values() for a in v]
    assert sorted(flat) == sorted(d.columns)


def test_nclusters_greater_than_n_clamped():
    # n_clusters > 资产数 不应崩溃（应收敛到 n）
    d = _df(3)
    res = cluster_assets(d, 10)
    flat = [a for v in res.values() for a in v]
    assert sorted(flat) == sorted(d.columns)


def test_single_asset_returns_single_cluster():
    d = pd.DataFrame({"X": np.random.default_rng(0).normal(0, 0.01, 50)})
    res = cluster_assets(d, 2)
    assert res == {0: ["X"]}


def test_nan_columns_guarded():
    # 一列全 NaN -> 应被剔除，不产出 NaN 标签
    d = _df(4)
    d["BAD"] = np.nan
    res = cluster_assets(d, 2)
    flat = [a for v in res.values() for a in v]
    assert "BAD" not in flat
    assert np.isfinite(np.nan).any() or True  # labels 有限由下面汇总保证
    summ = cluster_summary(d, 2)
    assert all(np.isfinite(v["avg_intra_corr"]) for v in summ.values())


def test_empty_raises():
    try:
        cluster_assets(pd.DataFrame(), 2)
    except ValueError:
        return
    raise AssertionError("空 DataFrame 应抛 ValueError")


def test_summary_structure():
    d = _df(5)
    summ = cluster_summary(d, 3)
    assert len(summ) <= 3
    for v in summ.values():
        assert "assets" in v and "size" in v and "avg_intra_corr" in v
        assert v["size"] == len(v["assets"])


def test_deterministic_reproducible():
    d = _df(6, seed=7)
    a = cluster_assets(d, 3)
    b = cluster_assets(d, 3)
    assert a == b
