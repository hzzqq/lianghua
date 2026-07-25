"""蒙特卡洛稳健性模拟：对回测权益曲线做重采样，评估策略稳健性。

- bootstrap：对历史日收益做**有放回块重采样**（保持短期相关性）。
- parametric：用历史均值/方差的正态分布重采样。
- simulate_paths：块重采样生成多条权益路径，输出逐日分位带(performance cone)。

输出收益分布、置信区间、最大回撤分布、亏损概率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..perf.metrics import max_drawdown

__all__ = ["bootstrap", "parametric", "simulate_paths"]


def _validate_equity(equity, init_cash):
    """统一校验输入，返回 (价值数组, 初始资金)。"""
    if not isinstance(equity, pd.Series):
        equity = pd.Series(np.asarray(equity, dtype=float))
    vals = equity.to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("equity 含非有限值(NaN/inf)，无法做蒙特卡洛模拟")
    if len(vals) < 2:
        raise ValueError("equity 至少需要 2 个观测点才能计算收益序列")
    if init_cash is None:
        init_cash = float(vals[0])
    if not np.isfinite(init_cash) or init_cash <= 0:
        raise ValueError(f"init_cash 必须为正数，收到 {init_cash!r}")
    return vals, init_cash


def _returns_from(vals: np.ndarray) -> np.ndarray:
    rets = pd.Series(vals).pct_change().dropna().to_numpy()
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        raise ValueError("收益序列为空(equity 缺少可计算收益的点)")
    return rets


def _block_resample(rets: np.ndarray, n_days: int, block: int, rng) -> np.ndarray:
    out = np.empty(n_days, dtype=float)
    pos = 0
    while pos < n_days:
        b = min(block, n_days - pos)
        start = rng.integers(0, max(1, len(rets) - b + 1))
        out[pos:pos + b] = rets[start:start + b]
        pos += b
    return out


def _run_sims(rets, init_cash, n, n_days, seed, sampler) -> tuple[np.ndarray, np.ndarray]:
    # 关键修复：把带 seed 的 Generator 贯穿整条模拟链路，
    # 原实现用 np.random.seed() 但 default_rng() 不读取全局状态 → seed 完全失效。
    rng = np.random.default_rng(seed)
    finals = np.empty(n, dtype=float)
    max_dds = np.empty(n, dtype=float)
    for i in range(n):
        sampled = sampler(rets, n_days, rng)
        eq = init_cash * np.cumprod(1.0 + sampled)
        finals[i] = eq[-1] / init_cash - 1.0
        max_dds[i] = max_drawdown(pd.Series(eq))
    return finals, max_dds


def bootstrap(equity: pd.Series, n: int = 1000, block: int = 5,
              init_cash: float | None = None, seed: int = 42) -> dict:
    """块 Bootstrap 重采样。"""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n 必须为正整数，收到 {n!r}")
    if not isinstance(block, int) or block < 1:
        raise ValueError(f"block 必须为正整数(否则分块循环死锁)，收到 {block!r}")
    vals, init_cash = _validate_equity(equity, init_cash)
    rets = _returns_from(vals)
    n_days = len(rets)
    finals, max_dds = _run_sims(
        rets, init_cash, n, n_days, seed,
        lambda r, d, rng: _block_resample(r, d, block, rng),
    )
    return _summarize(finals, max_dds, init_cash, n)


def parametric(equity: pd.Series, n: int = 1000, init_cash: float | None = None,
               seed: int = 42) -> dict:
    """参数法：正态重采样（用历史均值/标准差）。"""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n 必须为正整数，收到 {n!r}")
    vals, init_cash = _validate_equity(equity, init_cash)
    rets = _returns_from(vals)
    mu, sd = rets.mean(), rets.std(ddof=1)
    n_days = len(rets)
    finals, max_dds = _run_sims(
        rets, init_cash, n, n_days, seed,
        lambda r, d, rng: rng.normal(mu, sd, d),
    )
    return _summarize(finals, max_dds, init_cash, n)


def simulate_paths(equity: pd.Series, n: int = 500, block: int = 5,
                   init_cash: float | None = None, seed: int = 42,
                   percentiles: tuple = (5, 25, 50, 75, 95)) -> dict:
    """块重采样生成 n 条权益路径，返回逐日分位带(performance cone)。

    新能力：相比 bootstrap/parametric 只给终值分布，本函数给出
    整条路径的 p5/p25/p50/p75/p95 分位带，可直接绘制稳健性锥形区间。
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n 必须为正整数，收到 {n!r}")
    if not isinstance(block, int) or block < 1:
        raise ValueError(f"block 必须为正整数，收到 {block!r}")
    if not percentiles:
        raise ValueError("percentiles 不能为空")
    vals, init_cash = _validate_equity(equity, init_cash)
    rets = _returns_from(vals)
    n_days = len(rets)
    rng = np.random.default_rng(seed)
    paths = np.empty((n, n_days + 1), dtype=float)
    paths[:, 0] = init_cash
    for i in range(n):
        sampled = _block_resample(rets, n_days, block, rng)
        paths[i, 1:] = init_cash * np.cumprod(1.0 + sampled)
    bands = {int(p): np.percentile(paths, p, axis=0) for p in percentiles}
    final_ret = paths[:, -1] / init_cash - 1.0
    return {
        "n_simulations": n,
        "init_cash": init_cash,
        "days": np.arange(n_days + 1),
        "bands": bands,
        "median": bands.get(50, np.median(paths, axis=0)),
        "mean": paths.mean(axis=0),
        "final_return": {
            "mean": float(final_ret.mean()),
            "p5": float(np.percentile(final_ret, 5)),
            "p95": float(np.percentile(final_ret, 95)),
            "prob_loss": float((final_ret < 0).mean()),
        },
    }


def _summarize(finals: np.ndarray, max_dds: np.ndarray, init_cash: float, n: int) -> dict:
    return {
        "n_simulations": n,
        "init_cash": init_cash,
        "mean_return": float(finals.mean()),
        "median_return": float(np.median(finals)),
        "std_return": float(finals.std(ddof=1)) if n > 1 else 0.0,
        "p5": float(np.percentile(finals, 5)),
        "p95": float(np.percentile(finals, 95)),
        "best": float(finals.max()),
        "worst": float(finals.min()),
        "prob_loss": float((finals < 0).mean()),          # 亏损概率(相对初始)
        "mean_max_drawdown": float(max_dds.mean()),
        "worst_max_drawdown": float(max_dds.min()),
        "returns": finals,
        "max_drawdowns": max_dds,
    }
