"""参数优化：网格搜索 + 随机搜索 + Walk-forward 滚动验证。

设计为策略无关：调用方提供
  make_signals(df, **params) -> pd.Series    # 生成交易信号
  scorer(equity) -> float                    # 评分（默认夏普）
  run_backtest(df, signals) -> pd.Series(equity)  # 可选，缺省时构造买入持有 proxy
本模块负责遍历参数组合并在 walk-forward 下评估 OOS 表现。
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from ..perf.metrics import sharpe

__all__ = ["grid_search", "random_search", "walk_forward", "best_params", "summarize"]


def _coerce(v):
    """把 numpy 标量 / 整数型浮点转成 Python 原生类型，便于回传给策略函数。

    例：np.float64(3.0) -> int(3)；0.5 -> float(0.5)。
    pandas rolling 只接受整数 window，故整数参数必须转 int。
    """
    try:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (int, float, np.floating)):
            f = float(v)
            if f == int(f):
                return int(f)
            return f
    except (TypeError, ValueError):
        pass
    return v


def _proxy_equity(df: pd.DataFrame, sig: pd.Series) -> pd.Series:
    """run_backtest 缺省时的买入持有 proxy（按信号持仓，避免前视）。"""
    cols = list(getattr(df, "columns", []))
    price_col = "close" if "close" in cols else None
    if price_col is None:
        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num:
            raise ValueError("run_backtest 为 None 且 df 无价格列，无法构造 proxy")
        price_col = num[0]
    ret = df[price_col].pct_change().fillna(0.0)
    pos = sig.reindex(df.index).fillna(0.0).shift(1).fillna(0.0)
    return (1.0 + pos * ret).cumprod() * 100.0


def _score(make_signals, run_backtest, df, params, scorer):
    """单组参数的安全评估：失败隔离，返回 (score, error)。"""
    try:
        sig = make_signals(df, **params)
        eq = run_backtest(df, sig) if run_backtest else _proxy_equity(df, sig)
        return float(scorer(eq)), ""
    except Exception as e:  # 单组参数异常不得中止整轮搜索
        return float("nan"), repr(e)[:200]


def grid_search(
    df: pd.DataFrame,
    make_signals,
    param_grid: dict,
    scorer=None,
    run_backtest=None,
):
    """笛卡尔积遍历参数，返回按分数降序的参数表。

    run_backtest(df, signals) -> pd.Series(equity) 为可选；
    缺省时构造买入持有 proxy（按信号持仓，避免前视）。
    """
    scorer = scorer or sharpe
    if not param_grid:
        raise ValueError("param_grid 为空，无可搜索的参数组合")
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    if not combos:
        raise ValueError("param_grid 展开后无可搜索的参数组合")
    rows = []
    for vals in combos:
        params = {k: _coerce(v) for k, v in zip(keys, vals)}
        score, err = _score(make_signals, run_backtest, df, params, scorer)
        rows.append({**params, "score": score, "error": err})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def random_search(
    df: pd.DataFrame,
    make_signals,
    param_dist: dict,
    n_iter: int = 50,
    scorer=None,
    run_backtest=None,
    seed: int = 42,
):
    """随机采样参数空间（大空间远快于网格）。

    param_dist: 每项为
      - list/set: 从中均匀随机选取（如 [5, 10, 20]）
      - (lo, hi): 连续均匀采样
      - (lo, hi, "int"|"uniform"): 整型/连续采样
    """
    scorer = scorer or sharpe
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError(f"n_iter 必须为正整数，收到 {n_iter!r}")
    if not param_dist:
        raise ValueError("param_dist 为空，无可搜索的参数组合")
    rng = np.random.default_rng(seed)
    keys = list(param_dist.keys())
    rows = []
    for _ in range(n_iter):
        raw = {k: _sample(param_dist[k], rng) for k in keys}
        params = {k: _coerce(raw[k]) for k in keys}
        score, err = _score(make_signals, run_backtest, df, params, scorer)
        rows.append({**params, "score": score, "error": err})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _sample(v, rng):
    if isinstance(v, (list, set)):
        # 隐性修复：空 list/set 会让 rng.choice([]) 抛 ValueError 且不在 _score 隔离内
        if not v:
            raise ValueError("param_dist 含空 list/set，无法采样")
        return rng.choice(list(v))
    if isinstance(v, tuple):
        if len(v) == 2:
            lo, hi = v
            return float(rng.uniform(lo, hi))
        if len(v) == 3 and v[2] in ("int", "uniform"):
            lo, hi, kind = v
            if kind == "int":
                return int(rng.integers(lo, hi + 1))
            return float(rng.uniform(lo, hi))
    raise ValueError(f"param_dist 项 {v!r} 不支持；请用 list 或 (lo,hi)/(lo,hi,'int')")


def walk_forward(
    df: pd.DataFrame,
    make_signals,
    param_grid: dict,
    train_size: int = 250,
    test_size: int = 60,
    step: int | None = None,
    scorer=None,
    run_backtest=None,
):
    """滚动窗口：每窗训练集选最优参数，在测试集评估 OOS，汇总平均。

    返回 (oos_scores: list[float], best_params_per_window: list[dict])
    """
    scorer = scorer or sharpe
    if not isinstance(train_size, int) or train_size < 1:
        raise ValueError(f"train_size 必须为正整数，收到 {train_size!r}")
    if not isinstance(test_size, int) or test_size < 1:
        raise ValueError(f"test_size 必须为正整数，收到 {test_size!r}")
    step = step if step is not None else test_size
    if not isinstance(step, int) or step < 1:
        raise ValueError(f"step 必须为正整数，收到 {step!r}")
    oos_scores, best_params = [], []
    n = len(df)
    i = 0
    while i + train_size + test_size <= n:
        train = df.iloc[i:i + train_size]
        test = df.iloc[i + train_size:i + train_size + test_size]
        grid = grid_search(train, make_signals, param_grid, scorer, run_backtest)
        if grid.empty or pd.isna(grid.iloc[0]["score"]):
            i += step
            continue
        best = {k: _coerce(grid.iloc[0][k]) for k in param_grid}
        best_params.append(best)
        sig = make_signals(test, **best)
        eq = run_backtest(test, sig) if run_backtest else _proxy_equity(test, sig)
        oos_scores.append(float(scorer(eq)))
        i += step
    return oos_scores, best_params


def best_params(res: pd.DataFrame) -> dict | None:
    """从 grid_search / random_search 结果中取综合分最高的参数组合。

    隐性修复：当全部组合均失败时 score 为 NaN，sort_values(ascending=False)
    会把 NaN 排到末尾，iloc[0] 会取到 NaN 组合；这里只取有效（非 NaN）首行。
    """
    if res is None or res.empty:
        return None
    valid = res[~res["score"].isna()]
    if valid.empty:
        return None
    row = valid.iloc[0]
    keys = [c for c in res.columns if c not in ("score", "error")]
    return {k: row[k] for k in keys}


def summarize(res: pd.DataFrame) -> dict:
    """汇总一次搜索结果（新增能力）：最佳分、有效/失败数、分数均值与标准差。

    供调用方快速判断搜索质量与过拟合风险，无需逐行翻阅结果表。
    """
    if res is None or res.empty:
        return {"best_score": float("nan"), "mean": float("nan"),
                "std": float("nan"), "n_valid": 0, "n_fail": 0, "n_total": 0}
    scores = pd.to_numeric(res["score"], errors="coerce")
    valid = scores.dropna()
    fails = res.get("error", pd.Series([], dtype=object)).fillna("") != ""
    return {
        "best_score": float(valid.max()) if not valid.empty else float("nan"),
        "mean": float(valid.mean()) if not valid.empty else float("nan"),
        "std": float(valid.std()) if not valid.empty else float("nan"),
        "n_valid": int(len(valid)),
        "n_fail": int(fails.sum()),
        "n_total": int(len(res)),
    }
