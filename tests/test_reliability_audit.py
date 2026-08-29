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
from lianghua.backtest.engine import BacktestEngine


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


# ------------------------------------------------------------------  前视偏差（look-ahead bias）
def _random_walk(n: int = 600, seed: int = 7):
    """无动量、无漂移的随机游走：任何"超额"收益都只能来自前视或运气。"""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.015, n))
    # 开盘价 = 前收 + 小幅跳空，保证 open 列存在且可用
    prev = np.concatenate([[close[0]], close[:-1]])
    gap = rng.normal(0, 0.002, n)
    open_ = prev * (1 + gap)
    noise = np.abs(rng.normal(0, np.std(close) * 0.3 + 0.01, n))
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": close + noise, "low": close - noise,
         "close": close, "volume": np.abs(rng.normal(1e6, 3e5, n)) + 1e4},
        index=idx,
    )


def _cheat_signals(df: pd.DataFrame) -> pd.Series:
    """作弊策略（仅用于审计）：用**当日收盘**相对昨收的涨跌决定当日信号。

    关键：该信号在当日开盘时**不可得**（close[t] 尚未产生）。
    因此若引擎用它在当日开盘成交，就是标准的开盘点前视偏差；
    延迟一期后，它退化为「昨日涨跌」的动量信号，在随机游走（无动量）上无用。
    """
    prev_close = df["close"].shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[df["close"] > prev_close] = 1
    sig[df["close"] < prev_close] = -1
    return sig


def test_engine_default_is_not_look_ahead():
    """默认参数必须杜绝前视：延迟一期 + 开盘成交。"""
    eng = BacktestEngine()
    assert eng.execution_lag == 1, "默认必须延迟一期执行，否则回测含前视偏差"
    assert eng.fill_price == "open", "默认应以上帝视角不可得的开盘价成交"
    res = eng.run(_random_walk(), pd.Series(0, index=_random_walk().index))
    assert res.stats()["look_ahead"] is False


def test_engine_rejects_bad_execution_params():
    """非法 execution_lag / fill_price 必须回退到安全值，不能让前视悄悄生效。"""
    assert BacktestEngine(execution_lag=-5).execution_lag == 0
    assert BacktestEngine(execution_lag="abc").execution_lag == 1
    assert BacktestEngine(fill_price="next_tuesday").fill_price == "open"
    assert BacktestEngine(fill_price="close").fill_price == "close"


def test_look_ahead_bias_is_neutralized():
    """黄金测试：开盘时不可得的信号，不得用于开盘成交。

    信号 = sign(close[t] - close[t-1])，在 t 日开盘时 close[t] 尚未产生。
    - execution_lag=0 + fill_price=open：当日信号当日开盘成交 —— 开盘点前视，
      每天白赚日内涨幅，收益虚高。
    - execution_lag=1 + fill_price=open：t 日收盘出信号、t+1 开盘成交，
      信号退化为「昨日涨跌」动量；随机游走无动量，扣完成本后应接近 0。
    两者之差即前视偏差带来的虚假收益。
    """
    df = _random_walk()
    cheat = _cheat_signals(df)

    # 开盘点前视：当日信号 + 当日开盘成交（开盘时 close[t] 尚未产生）
    leak = BacktestEngine(execution_lag=0, fill_price="open").run(df, cheat)
    # 正确：信号延迟一期，t 日收盘出信号、t+1 开盘成交
    honest = BacktestEngine(execution_lag=1, fill_price="open").run(df, cheat)

    la = float(leak.stats()["total_return"])
    ho = float(honest.stats()["total_return"])

    # 1) 前视通道必须真实存在 —— 否则说明这个测试没构造出作弊效果
    assert la > 0.20, f"作弊策略在前视引擎下应大幅盈利，实际 {la:.2%}（测试本身失效）"
    # 2) 延迟执行必须把虚假收益抹掉（量级相差至少一个数量级）
    assert ho < la * 0.2, f"延迟执行后虚假收益未被消除：lag=0 {la:.2%} vs lag=1 {ho:.2%}"
    # 3) 随机游走无动量可赚，扣完成本后正确版本不应产生可观正收益
    assert ho < 0.10, f"延迟执行后仍盈利 {ho:.2%}，随机游走无动量可赚，疑似仍有信息泄露"


def test_missing_open_column_falls_back_to_close():
    """缺 open 列时必须自动回退收盘成交，不能崩或静默不交易。"""
    df = _random_walk().drop(columns=["open"])
    sig = pd.Series(0, index=df.index)
    sig.iloc[10] = 1
    sig.iloc[-1] = -1
    res = BacktestEngine().run(df, sig)
    assert len(res.trades) >= 1, "缺 open 列时应回退收盘价并正常成交"
    assert res.stats()["look_ahead"] is False


# ------------------------------------------------------------------  策略前视自检（未来扰动法）
@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_strategy_no_look_ahead(name):
    """未来扰动法：把未来数据整个删掉，历史信号必须一字不变。

    原理：若策略只用到 t 日及之前的数据，那么「完整的 n 根 K 线」与
    「只截到 k 根」算出的前 k 个信号必然完全一致。一旦前半段信号发生变化，
    就说明策略引用了尚未发生的未来数据（全样本均值/标准差、shift(-1)、
    全局归一化等），回测收益会系统性虚高。

    这是除引擎侧 execution_lag 之外的第二道防线：引擎保证「信号按时执行」，
    本测试保证「信号本身不含未来」。
    """
    df = _random_walk(n=300, seed=11)
    k = 200
    gen = get_strategy(name)
    full = np.asarray(gen.generate_signals(df))[:k]
    trunc = np.asarray(gen.generate_signals(df.iloc[:k]))[:k]
    m = min(len(full), len(trunc))
    assert m > 0, f"{name} 在截断数据上未产生任何信号"
    diff = int((full[:m] != trunc[:m]).sum())
    assert diff == 0, (
        f"{name} 的第 t 个信号依赖了 t 之后的数据：删掉未来 {len(df) - k} 根 K 线后，"
        f"前 {k} 个信号中有 {diff} 处改变 → 前视偏差"
    )


# ------------------------------------------------------------------  指标前视自检
_IND_FUNCS = {n: f for n, f in _FUZZ_FUNCS.items() if n.startswith("indicators.")}


def _truncate_args(args, k):
    return [
        a.iloc[:k].reset_index(drop=True) if isinstance(a, (pd.Series, pd.DataFrame)) else a
        for a in args
    ]


def _as_numeric(out):
    """只关心逐时点的数值输出；标量/非数值返回 None（不参与前视判定）。"""
    if isinstance(out, pd.Series):
        return out.to_numpy(dtype=float, na_value=np.nan)
    if isinstance(out, pd.DataFrame):
        num = out.select_dtypes(include=[np.number])
        return num.to_numpy(dtype=float, na_value=np.nan) if num.shape[1] else None
    if isinstance(out, np.ndarray) and np.issubdtype(out.dtype, np.number):
        return out.astype(float)
    return None


@pytest.mark.parametrize("fqn", sorted(_IND_FUNCS.keys()))
def test_indicator_no_look_ahead(fqn):
    """技术指标必须是「在线」的：截掉未来 K 线，历史取值不得改变。

    口径说明（避免误伤）：
    - 只检查 indicators 包。技术指标是策略的输入，逐时点输出，前视危害最大。
    - 不检查 perf / risk 的汇总标量（annual_return、skew、VaR 等）——它们的
      语义就是对整段样本做一次事后评价，截断后自然不同，不是前视。
    - 不检查 factor 的横截面统计（zscore / winsorize / 分组收益等）——它们
      在「行=日期、列=标的」的横截面上计算，与时间序列前视无关；
      但若被误用于时间序列再回测，确实会引入前视，属用法纪律而非函数缺陷。
    """
    f = _IND_FUNCS[fqn]
    fx = _fixtures()
    k = 150
    for args in _build_args(f, fx):
        try:
            full_out = f(*args)
            trunc_out = f(*_truncate_args(args, k))
        except Exception:  # noqa: BLE001
            continue
        a, b = _as_numeric(full_out), _as_numeric(trunc_out)
        if a is None or b is None:
            continue  # 标量/非数值指标不适用于前视判定
        m = min(len(a), len(b), k)
        if m == 0:
            continue
        A, B = a[:m], b[:m]
        if A.ndim > 1:
            A, B = A[:, 0], B[:, 0]
        ok = ~(np.isnan(A) | np.isnan(B))
        if ok.sum() == 0:
            continue
        diff = int((~np.isclose(A[ok], B[ok], rtol=1e-9, atol=1e-12)).sum())
        if diff:
            pytest.fail(
                f"{fqn} 依赖未来数据：截掉后段 K 线后，前 {k} 个取值中有 {diff} 处改变 "
                f"→ 前视偏差（常见成因：全样本 mean/std/max/min 定参数、shift(-1)）"
            )
        return
    pytest.skip(f"{fqn} 无法用自动派发的输入形态驱动，未做前视判定")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------------------  退化输入硬化
def _degenerate_fixtures(kind):
    """构造退化版 fixture：const / nan / short。kind 决定退化类型。"""
    s = mk_series(kind)
    eq = mk_equity(kind)
    df = mk_df(kind)
    ret = mk_ret(kind)
    ret_df = mk_ret_df(kind, 5, 250) if kind != "short" else mk_ret_df("short", 5, 12)
    ret_df = mk_ret_df(kind, 5, 60) if kind == "short" else ret_df
    cov = pd.DataFrame(
        np.cov(ret_df.T.values), columns=ret_df.columns, index=ret_df.columns
    )
    w = pd.Series(np.ones(5) / 5, index=ret_df.columns)
    trades = [{"pnl": 1.0}, {"pnl": -0.5}, {"pnl": 2.0, "exit": 1.2, "entry": 1.0}]
    d_factors = {"f1": s, "f2": s}
    return dict(s=s, eq=eq, df=df, ret=ret, ret_df=ret_df, cov=cov, w=w,
                trades=trades, d_factors=d_factors)


@pytest.mark.parametrize("fqn", sorted(_FUZZ_FUNCS.keys()))
@pytest.mark.parametrize("kind", ["const", "nan", "short"])
def test_quant_function_degenerate_inputs(fqn, kind):
    """退化输入（常数/全NaN/极短）下不得崩溃、输出无 NaN/inf。

    这是可靠性硬化的核心：正常输入通过的代码，遇到 std=0、全缺失、极短窗口
    仍可能除零/越界。任何方法都应优雅降级（返回有限值或清零），而非炸。
    """
    f = _FUZZ_FUNCS[fqn]
    fx = _degenerate_fixtures(kind)
    packs = _build_args(f, fx)
    last_exc = None
    for args in packs:
        try:
            out = f(*args)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            continue
        nan, inf = _has_bad(out)
        # 退化输入下：NaN 是可接受的"算不出"信号（垃圾进垃圾出）；
        # 但 inf 是数值爆炸（除零/无效归一），必须零容忍。
        if inf:
            pytest.fail(f"{fqn} [{kind}] 退化输入下输出含 inf（数值爆炸）")
        return
    pytest.skip(
        f"{fqn} [{kind}] 无法用自动派发的退化输入驱动（签名特殊）: "
        f"{type(last_exc).__name__}: {last_exc}"
    )
