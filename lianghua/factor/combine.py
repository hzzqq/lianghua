"""因子合成与稳定性（迭代 130）：factor_combine / factor_autocorr。

纯 pandas/numpy，零额外依赖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def factor_combine(factors: dict, weights: dict | None = None,
                   method: str = "zscore", clip: float | None = None) -> pd.Series:
    """多因子合成为单一综合因子。

    参数
    ----
    factors : {name: pd.Series}  各因子（同索引）
    weights : {name: float}      各因子权重，缺省等权
    method  : "zscore"（标准化后加权）| "rank"（排名百分位后加权）| "minmax"（线性归一化到[0,1]）
    clip    : 0~0.5 的缩尾分位（如 0.01 表示两端各截断 1%），合成前先 winsorize 抑制极端值

    返回
    ----
    综合因子 pd.Series（与输入同索引）
    """
    if not factors:
        raise ValueError("factors 不能为空")
    names = list(factors.keys())
    if weights is not None:
        unknown = [k for k in weights if k not in names]
        if unknown:
            raise ValueError(f"weights 含未知因子（不在 factors 中）：{unknown}")
    if clip is not None and not (0 <= clip < 0.5):
        raise ValueError("clip 必须在 [0, 0.5) 区间")
    w = weights or {n: 1.0 / len(names) for n in names}
    df = pd.DataFrame({n: pd.Series(factors[n]).astype(float) for n in names})
    df = df.replace([np.inf, -np.inf], np.nan)
    all_na = df.isna().all()
    if all_na.any():
        bad = df.columns[all_na].tolist()
        raise ValueError(f"以下因子全为 NaN，无法合成：{bad}")
    df = df.fillna(df.mean())  # 局部缺失用列均值轻量填补
    if clip is not None:
        lo = df.quantile(clip)
        hi = df.quantile(1 - clip)
        df = df.clip(lower=lo, upper=hi, axis=1)
    if method == "rank":
        norm = df.rank(pct=True)
    elif method == "minmax":
        lo = df.min()
        hi = df.max()
        span = (hi - lo).replace(0, 1.0)
        norm = (df - lo) / span
    else:  # zscore
        norm = (df - df.mean()) / (df.std(ddof=0) + 1e-12)
    wv = pd.Series({n: float(w.get(n, 0.0)) for n in names})
    sw = wv.sum()
    if sw == 0:
        raise ValueError("权重之和为零，无法合成（检查 weights）")
    wv = wv / sw
    combo = (norm * wv).sum(axis=1)
    return combo.rename("combined_factor")


def factor_autocorr(factor_panel: pd.DataFrame, lag: int = 1) -> float:
    """因子自相关（换手代理）：相邻期截面因子的秩相关均值，越高越稳定/低换手。

    参数
    ----
    factor_panel : index=日期, columns=资产 的因子面板
    lag          : 相邻期滞后
    """
    fp = factor_panel.astype(float)
    corrs = []
    for i in range(lag, len(fp)):
        a = fp.iloc[i - lag].rank()
        b = fp.iloc[i].rank()
        c = a.corr(b)
        if pd.notna(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def factor_neutrality(factor, protect) -> pd.Series | pd.DataFrame:
    """因子中性化：对 protect 做横截面 OLS 正交化，剔除其线性影响（如市值/行业哑变量）。

    参数
    ----
    factor  : 单期因子（一维 Series，索引=资产）或面板（index=日期,columns=资产）
    protect : 与 factor 同形——单期为 1-D Series（同资产索引）；面板为同形 DataFrame

    返回
    ----
    残差因子（同形）
    """
    f = factor.astype(float)

    def _cross_section(y: pd.Series, X) -> pd.Series:
        if X is None:
            return y
        if isinstance(X, pd.Series):
            if X.isna().all():
                return y
            Xd = np.column_stack([np.ones(len(y)), X.values])
        else:  # DataFrame 多变量
            if X.shape[1] == 0:
                return y
            Xd = np.column_stack([np.ones(len(y)), X.values])
        yv = y.values
        try:
            beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
        except np.linalg.LinAlgError:
            return y
        resid = yv - Xd @ beta
        return pd.Series(resid, index=y.index)

    if isinstance(f, pd.DataFrame):
        prot = protect.astype(float).reindex(index=f.index, columns=f.columns)
        out = f.copy()
        for d in f.index:
            Xd_row = prot.loc[d] if (prot is not None and d in prot.index) else None
            out.loc[d] = _cross_section(f.loc[d], Xd_row)
        return out.rename(columns=lambda c: f"{c}_neutral")
    # 一维
    prot = protect.astype(float).reindex(f.index)
    return _cross_section(f, prot).rename("neutral_factor")


def factor_turnover(factor) -> float:
    """因子换手率：逐期因子排序位移的均值（越高越频繁换仓），结果∈[0,1]。

    面板(DataFrame: 行=日期,列=资产)：各期排名位移的平均占比（相对满秩）。
    一维 Series(按时间)：相邻期变化绝对值 / 自身绝对值 的均值。
    """
    fp = pd.DataFrame(factor).astype(float)
    if fp.shape[1] == 1:
        s = fp.iloc[:, 0]
        diffs = s.diff().abs()
        denom = s.abs().replace(0, np.nan)
        to = (diffs / denom).mean(skipna=True) if denom.notna().any() else 0.0
        return float(min(max(0.0, to), 1.0))
    # 面板：平均排名位移 / 资产数 → 每步平均移动了满秩的比例
    ranks = fp.rank()
    diffs = ranks.diff().abs().iloc[1:]
    if len(diffs) == 0:
        return 0.0
    to = diffs.values.mean() / max(fp.shape[1], 1)
    return float(min(max(0.0, to), 1.0))


def factor_icir(factor, forward_returns=None) -> float:
    """因子信息系数 IR（Information Ratio of IC）。

    IC = 因子值与前瞻收益的截面秩相关；IR = mean(IC) / std(IC)（越高越稳定有效）。

    参数
    ----
    factor          : 因子面板（index=日期, columns=资产）
    forward_returns : 同期前瞻收益面板（与 factor 同形）；缺省时以「下期因子值」
                      作为前瞻收益的代理（自回归近似）。

    返回
    ----
    ICIR 标量（float）
    """
    fp = factor.astype(float)
    if forward_returns is None:
        fwd = fp.shift(-1)
    else:
        fwd = pd.DataFrame(forward_returns).astype(float).reindex(
            index=fp.index, columns=fp.columns)
    ics = []
    for i in range(1, len(fp)):
        a = fp.iloc[i - 1].rank()
        b = fwd.iloc[i - 1].rank()
        c = a.corr(b)
        if pd.notna(c):
            ics.append(float(c))
    if len(ics) < 2:
        return 0.0
    arr = np.array(ics)
    sd = arr.std(ddof=0)
    if sd <= 0:
        return 0.0
    return float(arr.mean() / sd)


def factor_rank_ic(factor, forward_returns=None) -> float:
    """因子秩 IC（Information Coefficient）均值：因子值与前瞻收益的截面秩相关平均。

    单一标量；>0 表示因子方向与收益正相关、具备选股能力。
    """
    fp = factor.astype(float)
    if forward_returns is None:
        fwd = fp.shift(-1)
    else:
        fwd = pd.DataFrame(forward_returns).astype(float).reindex(
            index=fp.index, columns=fp.columns)
    ics = []
    for i in range(1, len(fp)):
        a = fp.iloc[i - 1].rank()
        b = fwd.iloc[i - 1].rank()
        c = a.corr(b)
        if pd.notna(c):
            ics.append(float(c))
    return float(np.mean(ics)) if ics else 0.0


def factor_decay(factor, forward_returns=None, lags=(1, 5, 10, 20)) -> pd.Series:
    """因子 IC 衰减：在多个前瞻滞后 lag 上分别计算秩 IC，观察因子有效期。

    返回以 lag 为索引的 IC 序列；衰减越快说明因子时效性越短。
    """
    fp = factor.astype(float)
    if forward_returns is None:
        fwd = fp.shift(-1)
    else:
        fwd = pd.DataFrame(forward_returns).astype(float).reindex(
            index=fp.index, columns=fp.columns)
    out = {}
    for L in lags:
        ics = []
        for i in range(1, len(fp) - L + 1):
            a = fp.iloc[i - 1].rank()
            j = i - 1 + L - 1
            if j >= len(fwd):
                break
            b = fwd.iloc[j].rank()
            c = a.corr(b)
            if pd.notna(c):
                ics.append(float(c))
        out[L] = float(np.mean(ics)) if ics else 0.0
    return pd.Series(out, name="ic_decay")


def factor_winsorize(factor, lower: float = 0.01, upper: float = 0.99):
    """因子缩尾：把极端值截断到分位边界，抑制异常值对合成/IC 的污染。

    支持一维 Series（按自身分位）或面板 DataFrame（按列分位）。
    """
    fp = factor.astype(float)
    if isinstance(fp, pd.Series):
        return fp.clip(lower=fp.quantile(lower), upper=fp.quantile(upper))
    out = fp.copy()
    for c in out.columns:
        lo, hi = out[c].quantile(lower), out[c].quantile(upper)
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


# ============================================================
# 迭代 181+：更多因子稳定性 / 组合函数（超自驱动目标）
# ============================================================
def factor_corr(factor, target) -> float:
    """因子与目标（如前瞻收益）的截面相关均值：>0 同向、越稳定越有效。"""
    fa = pd.DataFrame(factor).astype(float)
    tg = pd.DataFrame(target).astype(float)
    if fa.shape[1] == 1 and tg.shape[1] > 1:
        fa = pd.concat([fa] * tg.shape[1], axis=1)
    if tg.shape[1] == 1 and fa.shape[1] > 1:
        tg = pd.concat([tg] * fa.shape[1], axis=1)
    fa = fa.reindex(columns=tg.columns)
    corrs = []
    for i in range(len(fa)):
        c = fa.iloc[i].corr(tg.iloc[i])
        if pd.notna(c):
            corrs.append(float(c))
    return float(np.mean(corrs)) if corrs else 0.0


def factor_orthogonalize(factor_a, factor_b) -> pd.DataFrame:
    """因子正交化：把 a 对 b 做横截面回归取残差，剔除 b 的线性影响。"""
    a = pd.DataFrame(factor_a).astype(float)
    b = pd.DataFrame(factor_b).astype(float).reindex(index=a.index, columns=a.columns)
    out = a.copy()
    for d in a.index:
        x = b.loc[d].values.astype(float)
        y = a.loc[d].values.astype(float)
        if x.std() == 0:
            continue
        beta, *_ = np.polyfit(x, y, 1)
        out.loc[d] = y - beta * x
    return out


def factor_weight_decay(factor_panel, halflife: int = 20) -> pd.Series:
    """因子时间衰减加权：越近期权重越高，输出每资产的时间衰减综合因子。"""
    fp = pd.DataFrame(factor_panel).astype(float)
    n = len(fp)
    if n == 0:
        return pd.Series(dtype=float)
    w = 0.5 ** (np.arange(n)[::-1] / max(halflife, 1))
    w = w / w.sum()
    weighted = (fp.mul(w, axis=0)).sum(axis=0)
    return weighted.rename("decayed_factor")


def factor_cross_section(factor_panel) -> pd.DataFrame:
    """因子横截面标准化：每日每行去均值除标准差（z-score），便于跨期可比。"""
    fp = pd.DataFrame(factor_panel).astype(float)
    vals = fp.values
    mu = vals.mean(axis=1, keepdims=True)
    sd = vals.std(axis=1, keepdims=True)
    z = (vals - mu) / (sd + 1e-12)
    out = pd.DataFrame(z, index=fp.index, columns=fp.columns)
    return out.rename(columns=lambda c: f"{c}_z")


def factor_portfolio(factor_panel, top: float = 0.2) -> pd.Series:
    """因子多空组合权重：取末态截面前 top 多头、后 top 空头（等权）。"""
    fp = pd.DataFrame(factor_panel).astype(float)
    if fp.empty:
        return pd.Series(dtype=float)
    last = fp.iloc[-1]
    n = len(last)
    k = max(1, int(round(n * top)))
    w = pd.Series(0.0, index=last.index)
    order = last.sort_values(ascending=False)
    w[order.index[:k]] = 1.0 / k
    w[order.index[-k:]] = -1.0 / k
    return w.rename("ls_weights")


def factor_decay_halflife(ic_decay) -> float:
    """因子 IC 衰减半衰期：由 IC 衰减序列（按 lag 索引）对数拟合估计。
    返回半衰期（期数）；不衰减返回 inf。"""
    s = pd.Series(ic_decay).astype(float)
    s = s[s.abs() > 1e-9]
    if len(s) < 2:
        return float("inf")
    x = np.arange(len(s))
    y = np.log(s.abs().values + 1e-12)
    beta, *_ = np.polyfit(x, y, 1)
    if beta >= 0:
        return float("inf")
    return float(-np.log(2) / beta)
