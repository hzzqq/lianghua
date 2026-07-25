"""简化 Brinson 归因：配置效应 + 选择效应 + 交叉项，并支持分组聚合。

- brinson_attribution : 逐资产分解（配置/选择/交叉/总效应）。
- attribution_summary : 聚合汇总 + 恒等式校验（总效应 == 主动收益）。
- sector_attribution  : 按行业/分组聚合归因（新增可观测能力）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["brinson_attribution", "attribution_summary", "sector_attribution"]


def _align(port_w, bench_w, asset_ret_p, asset_ret_b):
    """统一校验并截取共同资产子集。

    隐性修复：原实现对空交集静默返回空 DataFrame（误以为归因成功），
    且对 NaN/inf 输入不做任何校验，污染后续加权求和。
    """
    for name, s in (("port_w", port_w), ("bench_w", bench_w),
                    ("asset_ret_p", asset_ret_p), ("asset_ret_b", asset_ret_b)):
        if s is not None and not isinstance(s, pd.Series):
            raise TypeError(f"{name} 必须是 pandas.Series")
    if asset_ret_b is None:
        asset_ret_b = asset_ret_p
    common = list(set(port_w.index) & set(bench_w.index)
                  & set(asset_ret_p.index) & set(asset_ret_b.index))
    if not common:
        raise ValueError("组合/基准/收益之间无共同资产，无法归因")
    pw = port_w[common].astype(float)
    bw = bench_w[common].astype(float)
    rp = asset_ret_p[common].astype(float)
    rb = asset_ret_b[common].astype(float)
    for name, s in (("port_w", pw), ("bench_w", bw),
                    ("asset_ret_p", rp), ("asset_ret_b", rb)):
        if not np.isfinite(s.to_numpy()).all():
            raise ValueError(f"{name} 含 NaN/inf，无法归因")
    return pw, bw, rp, rb


def brinson_attribution(port_w: pd.Series, bench_w: pd.Series,
                        asset_ret_p: pd.Series,
                        asset_ret_b: pd.Series | None = None) -> pd.DataFrame:
    """逐资产 Brinson 归因。

    pw/bw：资产权重；rp/rb：组合/基准下各资产收益（rb 缺省沿用 rp）。
    """
    pw, bw, rp, rb = _align(port_w, bench_w, asset_ret_p, asset_ret_b)
    alloc = (pw - bw) * rb
    sel = bw * (rp - rb)
    inter = (pw - bw) * (rp - rb)
    total = pw * rp - bw * rb
    return pd.DataFrame({
        "配置效应": alloc, "选择效应": sel,
        "交叉项": inter, "总效应": total,
    })


def attribution_summary(port_w: pd.Series, bench_w: pd.Series,
                        asset_ret_p: pd.Series,
                        asset_ret_b: pd.Series | None = None) -> dict:
    """聚合归因汇总 + 恒等式校验（新增可观测能力）。

    返回配置/选择/交叉/总效应之和，主动收益，以及恒等式残差
    （总效应 - 主动收益，应约等于 0，用于自检归因一致性）。
    """
    df = brinson_attribution(port_w, bench_w, asset_ret_p, asset_ret_b)
    rb = asset_ret_b if asset_ret_b is not None else asset_ret_p
    idx = df.index
    active = float((port_w[idx] * asset_ret_p[idx]).sum()
                   - (bench_w[idx] * rb[idx]).sum())
    total = float(df["总效应"].sum())
    return {
        "配置效应": float(df["配置效应"].sum()),
        "选择效应": float(df["选择效应"].sum()),
        "交叉项": float(df["交叉项"].sum()),
        "总效应": total,
        "主动收益": active,
        "恒等式残差": total - active,
    }


def sector_attribution(port_w: pd.Series, bench_w: pd.Series,
                       asset_ret_p: pd.Series, groups: dict,
                       asset_ret_b: pd.Series | None = None) -> pd.DataFrame:
    """按行业/分组聚合 Brinson 归因（新增能力）。

    groups: {资产: 行业名}；返回每个分组的配置/选择/交叉/总效应合计。
    """
    if not isinstance(groups, dict):
        raise TypeError("groups 必须是 {资产: 行业} 映射")
    df = brinson_attribution(port_w, bench_w, asset_ret_p, asset_ret_b)
    sec = pd.Series(groups)
    sec = sec.reindex(df.index).dropna()
    if sec.empty:
        raise ValueError("groups 与归因资产无交集，无法分组")
    return df.loc[sec.index].groupby(sec.values).sum()
