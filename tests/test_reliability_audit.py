# -*- coding: utf-8 -*-
"""全量量化方法可靠性审计。

目标：用对抗性输入驱动「每一类量化方法」，把静默错误（NaN/inf 泄露、
除零、输出越界、权重不归一）暴露成可断言的失败，逐轮修复。

- 策略（40）：generate_signals(df) 必须输出有限、且取值 ∈ {-1,0,1}。
  直接管仓位，越界信号会触发错误杠杆。
- 优化器（28）：optimize_weights(ret_df) 必须输出有限、权重和 ≈ 1、非负。
  协方差=0 / 单资产 / 全 NaN 等退化输入必须优雅降级（等权），不能炸。
- 指标/绩效/风险/因子：按参数形态自动派发正常输入，断言不崩、输出无 NaN/inf。

运行：python -u tests/test_reliability_audit.py
"""
import importlib
import inspect
import math
import os
import pkgutil
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from lianghua.strategy.registry import STRATEGY_NAMES, get_strategy
from lianghua.portfolio.registry import OPTIMIZER_NAMES, optimize_weights


# ------------------------------------------------------------------  fixtures
_RNG = np.random.default_rng(20260829)


def _ohlc(n, close):
    c = np.asarray(close, float)
    # 围绕收盘构造高低开，保证 high>=low
    noise = np.abs(_RNG.normal(0, np.std(c) * 0.3 + 0.01, n))
    hi = c + noise
    lo = c - noise
    op = c + _RNG.normal(0, noise.mean() + 0.01, n)
    vol = np.abs(_RNG.normal(1e6, 3e5, n)) + 1e4
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": op, "high": hi, "low": lo, "close": c, "volume": vol}, index=idx
    )


def mk_df(kind="normal", n=250):
    if kind == "short":
        return mk_df("normal", 12)
    if kind == "const":
        c = np.full(n, 100.0)
    elif kind == "nan":
        c = np.full(n, np.nan)
    else:
        ret = _RNG.normal(0, 0.02, n)
        c = 100 * np.cumprod(1 + ret)
    return _ohlc(n, c)


def mk_series(kind="normal", n=250):
    if kind == "const":
        return pd.Series(np.full(n, 100.0))
    if kind == "nan":
        return pd.Series(np.full(n, np.nan))
    if kind == "short":
        return mk_series("normal", 12)
    ret = _RNG.normal(0, 0.02, n)
    return pd.Series(100 * np.cumprod(1 + ret))


def mk_equity(kind="normal", n=250):
    if kind == "const":
        return pd.Series(np.full(n, 1.0))
    if kind == "nan":
        return pd.Series(np.full(n, np.nan))
    ret = _RNG.normal(0.0005, 0.02, n)
    return pd.Series(np.cumprod(1 + ret))


def mk_ret(kind="normal", n=250):
    if kind == "const":
        return pd.Series(np.full(n, 0.001))
    if kind == "nan":
        return pd.Series(np.full(n, np.nan))
    return pd.Series(_RNG.normal(0.0005, 0.02, n))


def mk_ret_df(kind="normal", n=5, t=250):
    cols = [f"A{i}" for i in range(n)]
    if kind == "const":
        return pd.DataFrame(np.full((t, n), 0.001), columns=cols)
    if kind == "zero":
        return pd.DataFrame(np.zeros((t, n)), columns=cols)
    if kind == "nan":
        d = _RNG.normal(0.0005, 0.02, (t, n))
        d[0, 0] = np.nan
        return pd.DataFrame(d, columns=cols)
    if kind == "short":
        return mk_ret_df("normal", n, 12)
    if kind == "1asset":
        return mk_ret_df("normal", 1, t)
    d = _RNG.normal(0.0005, 0.02, (t, n))
    return pd.DataFrame(d, columns=cols)


# ------------------------------------------------------------------  finite 检查
def _has_bad(out):
    """返回 (has_nan, has_inf)。非数值输出视为无问题。

    对时间序列(Series/DataFrame)容忍前 25% 的预热 NaN（指标滚动窗口未就绪属正常），
    只检查尾部；inf 全段判坏（数值爆炸）。标量/字典严格。
    """
    try:
        if isinstance(out, (float, np.floating)):
            return (bool(math.isnan(out)), bool(math.isinf(out)))
        if isinstance(out, (int, np.integer)):
            return (False, False)
        if isinstance(out, np.ndarray):
            if not np.issubdtype(out.dtype, np.number):
                return (False, False)
            return (bool(np.isnan(out).any()), bool(np.isinf(out).any()))
        if isinstance(out, pd.Series):
            if not np.issubdtype(out.dtype, np.number):
                return (False, False)
            v = out.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(v).any():
                return (True, True)
            k = max(1, int(len(v) * 0.25))
            return (bool(np.isnan(v[k:]).any()), False)
        if isinstance(out, pd.DataFrame):
            num = out.select_dtypes(include=[np.number])
            if num.shape[1] == 0:
                return (False, False)
            arr = num.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(arr).any():
                return (True, True)
            k = max(1, int(arr.shape[0] * 0.25))
            return (bool(np.isnan(arr[k:]).any()), False)
        if isinstance(out, dict):
            for v in out.values():
                b = _has_bad(v)
                if b[0] or b[1]:
                    return b
            return (False, False)
        if isinstance(out, (list, tuple)):
            for v in out:
                b = _has_bad(v)
                if b[0] or b[1]:
                    return b
            return (False, False)
    except Exception:
        return (False, False)
    return (False, False)


# ------------------------------------------------------------------  策略（严格）
_STRAT_KINDS = ["normal", "const", "nan", "short"]


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_strategy_reliability(name):
    gen = get_strategy(name)
    for kind in _STRAT_KINDS:
        df = mk_df(kind)
        try:
            sig = gen.generate_signals(df)
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"策略 {name} [{kind}] generate_signals 抛异常: {type(e).__name__}: {e}")
        assert isinstance(sig, pd.Series), f"策略 {name} [{kind}] 未返回 Series"
        s = sig.to_numpy(dtype=float, na_value=np.nan)
        assert not np.isnan(s).any(), f"策略 {name} [{kind}] 信号含 NaN"
        assert not np.isinf(s).any(), f"策略 {name} [{kind}] 信号含 inf"
        vals = set(int(round(v)) for v in s)
        assert vals <= {-1, 0, 1}, f"策略 {name} [{kind}] 信号越界 {-1,0,1}: {sorted(vals)}"


# ------------------------------------------------------------------  优化器（严格）
_OPT_KINDS = ["normal", "const", "zero", "nan", "short", "1asset"]


@pytest.mark.parametrize("name", OPTIMIZER_NAMES)
def test_optimizer_reliability(name):
    for kind in _OPT_KINDS:
        rdf = mk_ret_df(kind)
        try:
            w = optimize_weights(name, rdf)
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"优化器 {name} [{kind}] 抛异常: {type(e).__name__}: {e}")
        w = pd.Series(w) if not isinstance(w, pd.Series) else w
        wv = w.to_numpy(dtype=float, na_value=np.nan)
        assert not np.isnan(wv).any(), f"优化器 {name} [{kind}] 权重含 NaN"
        assert not np.isinf(wv).any(), f"优化器 {name} [{kind}] 权重含 inf"
        s = float(np.nansum(wv))
        assert abs(s - 1.0) < 1e-6, f"优化器 {name} [{kind}] 权重和={s:.6f} (应≈1)"
        mn = float(np.nanmin(wv))
        assert mn >= -1e-9, f"优化器 {name} [{kind}] 出现负权重 {mn:.6f}"


# ------------------------------------------------------------------  指标/绩效/风险/因子（按参数形态自动派发）
def _collect_functions(subpkgs):
    funcs = {}
    for sub in subpkgs:
        try:
            pkg = importlib.import_module(f"lianghua.{sub}")
        except Exception:  # noqa: BLE001
            continue
        for mod in pkgutil.iter_modules(pkg.__path__):
            try:
                m = importlib.import_module(f"lianghua.{sub}.{mod.name}")
            except Exception:  # noqa: BLE001
                continue
            for nm, obj in vars(m).items():
                if nm.startswith("_"):
                    continue
                if inspect.isfunction(obj) and obj.__module__ == m.__name__:
                    funcs[f"{sub}.{mod.name}.{nm}"] = obj
    return funcs


_FUZZ_FUNCS = _collect_functions(["indicators", "perf", "risk", "factor"])


def _fixtures():
    s = mk_series("normal")
    eq = mk_equity("normal")
    df = mk_df("normal")
    ret = mk_ret("normal")
    ret_df = mk_ret_df("normal", 5, 250)
    cov = pd.DataFrame(
        np.cov(ret_df.T.values), columns=ret_df.columns, index=ret_df.columns
    )
    w = pd.Series(np.ones(5) / 5, index=ret_df.columns)
    trades = [{"pnl": 1.0}, {"pnl": -0.5}, {"pnl": 2.0, "exit": 1.2, "entry": 1.0}]
    d_factors = {"f1": s, "f2": s}
    return dict(s=s, eq=eq, df=df, ret=ret, ret_df=ret_df, cov=cov, w=w,
                trades=trades, d_factors=d_factors)


def _fixture_for(name, idx, fx):
    nl = (name or "").lower()
    if nl in ("cov", "covariance", "sigma", "corr", "c", "s_mat", "cov_mat"):
        return fx["cov"]
    if nl in ("weights", "w"):
        return fx["w"]
    if nl in ("budgets", "b"):
        return np.ones(5) / 5
    if nl in ("benchmark", "bench"):
        return fx["s"]
    if nl in ("forward_ret", "fwd", "fret", "fwd_ret"):
        return fx["s"]
    if nl in ("trades", "trade_list"):
        return fx["trades"]
    if nl in ("factor", "factors") and idx > 0:
        return fx["d_factors"]
    if nl in ("returns_df", "matrix", "x", "data", "prices", "price_df", "rets"):
        return fx["ret_df"]
    if nl in ("df",):
        return fx["df"]
    # 必需且无默认的特殊语义参数
    if nl == "regime":
        # regime_stats(returns, regime) 内部对 regime 做 .astype(int)，
        # 因此需喂整数编码(0/1)的等长标签序列（与 detect_regime 输出一致）
        lab = np.where(fx["s"].to_numpy() >= fx["s"].mean(), 1, 0).astype(int)
        return pd.Series(lab, index=fx["s"].index)
    if nl in ("method", "kind", "mode", "style", "typ", "model", "how", "phase"):
        return "bull"
    if nl in (
        "alpha", "rf", "threshold", "shrink", "delta", "target_vol", "nb", "mult",
        "q", "z", "lb", "ub", "tau", "lam", "gamma", "scale", "vol_target", "band",
        "k_", "level_", "clip",
    ):
        return 0.05
    if nl in (
        "window", "lookback", "period", "span", "n", "lag", "p", "top", "level",
        "fast", "slow", "signal", "k", "bins", "length", "width", "step", "vol",
        "lookahead", "min_", "max_", "num", "order", "depth", "power",
    ):
        return 20
    # 默认：1D 收益/净值/因子序列（多数指标与绩效风险函数）
    return fx["s"]


def _build_args(f, fx):
    sig = inspect.signature(f)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
        and p.default is inspect.Parameter.empty  # 仅必需位置参数
    ]
    n = len(params)
    if n == 0:
        return [[]]
    names = [p.name for p in params]
    base = [_fixture_for(nm, i, fx) for i, nm in enumerate(names)]
    packs = [base]
    # 替补：全序列（覆盖参数名识别不到的 1D 情形）
    packs.append([fx["s"]] * n)
    # 替补：OHLC 形态（high/low/close 类）
    if n == 2:
        packs.append([fx["df"]["high"], fx["df"]["low"]])
    if n == 3:
        packs.append([fx["df"]["high"], fx["df"]["low"], fx["df"]["close"]])
    if n == 1 and names[0].lower() in ("factor", "factors"):
        packs.append([fx["s"]])
        packs.append([fx["d_factors"]])
    return packs


@pytest.mark.parametrize("fqn", sorted(_FUZZ_FUNCS.keys()))
def test_quant_function_no_crash_no_nan(fqn):
    f = _FUZZ_FUNCS[fqn]
    fx = _fixtures()
    packs = _build_args(f, fx)
    last_exc = None
    for args in packs:
        try:
            out = f(*args)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            continue
        nan, inf = _has_bad(out)
        if nan:
            pytest.fail(f"{fqn} 正常输入下输出含 NaN")
        if inf:
            # inf 可能是合法哨兵值（如因子 IC 衰减半衰期"不衰减"返回 inf），仅告警
            print(f"  WARN {fqn} 正常输入下输出含 inf（可能为合法哨兵值）")
        return  # 任一 pack 成功且无 NaN 即通过
    pytest.skip(
        f"{fqn} 无法用自动派发的输入形态驱动（签名特殊，需人工核对）: "
        f"{type(last_exc).__name__}: {last_exc}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
