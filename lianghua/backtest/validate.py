"""样本外验证：区分「策略真有效」与「参数拟合出来的好结果」。

回测收益高只说明「在这段历史数据上有效」。真正的检验是样本外表现：
用前段数据定参数、后段数据验证。过拟合的策略在样本外会大幅退化。

两个入口：
- ``split_oos(df, train_ratio)``：按时间切分（严格前瞻，不 shuffling）。
- ``out_of_sample_report(df, signal_fn, ...)``：一次性给出样本内/样本外
  收益、衰减率与判定，UI/CLI 可直接展示。

零重依赖：仅 pandas/numpy。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestEngine, BacktestResult


def split_oos(df: pd.DataFrame, train_ratio: float = 0.6):
    """按时间顺序切分样本内/样本外（金融数据不可随机打乱，会引入前视）。

    返回 ``(train_df, test_df)``。train_ratio 会被夹到 (0.2, 0.9)，
    保证两段都有足够的统计长度。
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df 必须是 DataFrame")
    r = float(train_ratio)
    if not np.isfinite(r):
        r = 0.6
    r = min(max(r, 0.2), 0.9)
    k = int(len(df) * r)
    k = max(1, min(k, len(df) - 1))
    return df.iloc[:k], df.iloc[k:]


def _run(df: pd.DataFrame, signals: pd.Series, engine_kwargs: dict | None) -> BacktestResult:
    eng = BacktestEngine(**(engine_kwargs or {}))
    return eng.run(df, signals)


def out_of_sample_report(
    df: pd.DataFrame,
    signal_fn,
    train_ratio: float = 0.6,
    engine_kwargs: dict | None = None,
) -> dict:
    """样本外验证报告。

    ``signal_fn`` 是 ``f(df) -> signals``，通常传 ``get_strategy(name).generate_signals``，
    也可以传用户自己的函数。参数固定不变，只切换数据段——这样才能反映
    「同一套参数在未见数据上的表现」。

    返回字段：
    - ``in_sample`` / ``out_sample``：两段的 stats()
    - ``decay``：收益衰减率 = 1 - out/in（in<=0 时为 None，不具可比性）
    - ``verdict``：``robust`` / ``degraded`` / ``overfit`` / ``no_edge`` / ``unknown``
    - ``note``：一句话结论
    """
    if len(df) < 60:
        return {
            "ok": False, "verdict": "unknown",
            "note": f"样本仅 {len(df)} 根 K 线，不足以做样本外验证（建议 >=120）",
            "in_sample": None, "out_sample": None, "decay": None,
        }

    train_df, test_df = split_oos(df, train_ratio)
    if len(test_df) < 20:
        return {
            "ok": False, "verdict": "unknown",
            "note": f"样本外仅 {len(test_df)} 根 K 线，不足以评估",
            "in_sample": None, "out_sample": None, "decay": None,
        }

    try:
        in_res = _run(train_df, signal_fn(train_df), engine_kwargs)
        out_res = _run(test_df, signal_fn(test_df), engine_kwargs)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False, "verdict": "unknown", "note": f"样本外验证失败：{type(e).__name__}: {e}",
            "in_sample": None, "out_sample": None, "decay": None,
        }

    s_in = in_res.stats()
    s_out = out_res.stats()
    r_in = float(s_in.get("total_return") or 0.0)
    r_out = float(s_out.get("total_return") or 0.0)

    if r_in > 0:
        decay = float(1.0 - r_out / r_in)
        if r_out <= 0:
            verdict = "overfit"
            note = (f"样本内 {r_in:.1%} → 样本外 {r_out:.1%}："
                    f"收益由正转负，典型的过拟合，不建议据此配置资金")
        elif decay > 0.7:
            verdict = "overfit"
            note = (f"样本内 {r_in:.1%} → 样本外 {r_out:.1%}："
                    f"衰减 {decay:.0%}，大部分收益来自拟合历史噪声")
        elif decay > 0.4:
            verdict = "degraded"
            note = (f"样本内 {r_in:.1%} → 样本外 {r_out:.1%}："
                    f"衰减 {decay:.0%}，仍有效但明显减弱，参数宜保守")
        else:
            verdict = "robust"
            note = (f"样本内 {r_in:.1%} → 样本外 {r_out:.1%}："
                    f"衰减 {decay:.0%}，样本外保持，稳健性较好")
    else:
        decay = None
        verdict = "no_edge"
        note = (f"样本内收益 {r_in:.1%} 非正，策略在历史区间就没有优势，"
                f"样本外 {r_out:.1%} 无参考价值")

    return {
        "ok": True, "verdict": verdict, "note": note,
        "in_sample": s_in, "out_sample": s_out, "decay": decay,
        "in_sample_sanity": in_res.sanity(), "out_sample_sanity": out_res.sanity(),
    }
