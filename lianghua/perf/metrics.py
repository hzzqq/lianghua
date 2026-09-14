"""绩效指标：夏普比率、最大回撤、年化收益、汇总、Calmar 比率、水下曲线。

所有函数带净值守卫：空序列、非数值（NaN/inf）、非正净值都会给出清晰错误，
避免 inf/NaN 静默污染摘要与下游报告。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _guard_equity(equity, require_positive: bool = False) -> pd.Series:
    if not isinstance(equity, pd.Series):
        try:
            equity = pd.Series(equity)
        except Exception:
            raise TypeError(f"equity 必须是 1-D array-like，收到 {type(equity).__name__}")
    s = equity.astype(float)
    if s.empty:
        raise ValueError("equity 不能为空")
    if not np.isfinite(s).all():
        raise ValueError("equity 含 NaN/inf，无法计算绩效指标")
    if require_positive and (s <= 0).any():
        raise ValueError("equity 必须全为正（净值/价格不能为 0 或负）")
    return s


def _guard_positive_base(s: pd.Series) -> None:
    if s.iloc[0] == 0:
        raise ValueError("equity 首项为 0，无法计算相对收益/比率")


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe(equity: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    if periods < 1:
        raise ValueError(f"periods 必须为正整数，收到 {periods!r}")
    s = _guard_equity(equity)
    r = daily_returns(s)
    # 净值含 0 时 pct_change 会产出 inf，必须先剔除非有限收益，
    # 否则 r.std() 触发无效值警告且可能返回 inf 污染下游
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float((r.mean() - rf / periods) / r.std(ddof=1) * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    s = _guard_equity(equity)
    roll_max = s.cummax()
    dd = s / roll_max - 1.0
    dd = dd[roll_max > 0]          # 运行最高为 0 时（净值全 0）跳过，避免 0/0=nan
    return float(dd.min()) if len(dd) else 0.0


def underwater(equity: pd.Series) -> pd.Series:
    """水下曲线（回撤序列），取值落在 [-1, 0]，供可视化与可观测性。"""
    s = _guard_equity(equity)
    roll_max = s.cummax()
    return s / roll_max - 1.0


def annual_return(equity: pd.Series, periods: int = 252) -> float:
    if periods < 1:
        raise ValueError(f"periods 必须为正整数，收到 {periods!r}")
    s = _guard_equity(equity)
    _guard_positive_base(s)
    base = s.iloc[-1] / s.iloc[0]
    if base <= 0:
        return -0.999999
    return float(base ** (periods / len(s)) - 1)


def calmar_ratio(equity: pd.Series, periods: int = 252) -> float:
    """Calmar 比率 = 年化收益 / |最大回撤|；回撤为 0 时返回 0。"""
    if periods < 1:
        raise ValueError(f"periods 必须为正整数，收到 {periods!r}")
    s = _guard_equity(equity)
    mdd = max_drawdown(s)
    if mdd == 0:
        return 0.0
    return float(annual_return(s, periods) / abs(mdd))


def summary(equity: pd.Series, trades: list[dict]) -> dict:
    """生成绩效摘要，供 UI 与报告使用。"""
    s = _guard_equity(equity)
    _guard_positive_base(s)
    return {
        "final_equity": float(s.iloc[-1]),
        "total_return": float(s.iloc[-1] / s.iloc[0] - 1),
        "annual_return": annual_return(s),
        "sharpe": sharpe(s),
        "max_drawdown": max_drawdown(s),
        "calmar": calmar_ratio(s),
        "num_trades": len(trades),
    }


def win_rate(trades: list[dict]) -> float:
    """成交胜率（按配对买卖粗略估算）。

    .. deprecated::
        该方法用相邻 ``cash_after`` 的单调变化近似胜率，对含止损平仓、资金约束
        导致的非单调现金曲线极不稳健（会系统性高估或低估）。请改用
        :func:`round_trip_win_rate`（严格按 BUY→SELL 配对计算真实回合胜率）。
        此处保留仅为向后兼容。
    """
    if not trades:
        return 0.0
    pnl = [float(t.get("cash_after", 0.0)) for t in trades]
    # 以 cash_after 单调变化近似：上涨笔数占比
    ups = sum(1 for i in range(1, len(pnl)) if pnl[i] > pnl[i - 1])
    return ups / max(1, len(pnl) - 1)


def round_trip_stats(trades: list[dict]) -> dict:
    """按 BUY→SELL 严格配对的逐笔回合统计（修复 win_rate 的近似缺陷）。

    回测引擎每轮持仓恰好一笔 BUY + 一笔 SELL（含止损平仓 reason="stop_loss"），
    故可顺序配对。每回合净收益 = 卖出所得(价×量 − 费−税) − 买入成本(价×量 + 费)。
    胜 = 净收益 > 0。返回胜率、平均盈利、平均亏损、盈利因子、总净收益与回合数。

    这是「高胜率」类策略唯一可信的胜率口径：win_rate 的 cash_after 近似在
    现金曲线非单调（止损、分批、资金约束）时会失真。
    """
    trips: list[float] = []
    entry: tuple[float, float, float] | None = None
    for t in trades:
        side = t.get("side")
        if side == "BUY":
            entry = (float(t["price"]), float(t["qty"]), float(t.get("fee", 0.0)))
        elif side == "SELL" and entry is not None:
            bp, bq, bf = entry
            sp = float(t["price"]); sq = float(t["qty"]); sf = float(t.get("fee", 0.0))
            buy_cost = bp * bq + bf
            sell_pro = sp * sq - sf
            trips.append(sell_pro - buy_cost)
            entry = None
    if not trips:
        return {"n": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "profit_factor": 0.0, "total_pnl": 0.0}
    wins = [p for p in trips if p > 0]
    losses = [p for p in trips if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    return {
        "n": len(trips),
        "win_rate": (len(wins) / len(trips)) if trips else 0.0,
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (gross_loss / len(losses)) if losses else 0.0,
        "profit_factor": pf,
        "total_pnl": (gross_win - gross_loss),
    }


def round_trip_win_rate(trades: list[dict]) -> float:
    """逐笔配对胜率（真实口径）。详见 :func:`round_trip_stats`。"""
    return round_trip_stats(trades)["win_rate"]


def buyhold_equity(df: pd.DataFrame, init_cash: float = 1_000_000.0) -> pd.Series:
    """买入持有基准净值（用于对比）。

    自动选择价格列：优先 close（股票/基金/期货），
    期权则回退到 underlying（标的）或 option_price（权利金）。
    """
    for col in ("close", "underlying", "option_price"):
        if col in df.columns:
            price = df[col].astype(float)
            if not np.isfinite(price).all() or (price <= 0).any():
                raise ValueError(f"基准价格列 {col} 含非正/非有限值")
            ratio = price / price.iloc[0]
            return init_cash * ratio
    raise KeyError("df 中找不到可用价格列(close/underlying/option_price)")


def report(equity: pd.Series, trades: list[dict], df: pd.DataFrame | None = None,
           init_cash: float = 1_000_000.0) -> dict:
    """扩展绩效报告：含基准对比与胜率。"""
    s = _guard_equity(equity)
    rep = summary(s, trades)
    rep["win_rate"] = round_trip_win_rate(trades)
    if df is not None and len(df) == len(s):
        bench = buyhold_equity(df, init_cash)
        rep["benchmark_return"] = float(bench.iloc[-1] / bench.iloc[0] - 1)
        excess = rep["total_return"] - rep["benchmark_return"]
        rep["excess_return"] = float(excess)
    return rep


def attribution(equity_by_asset: dict[str, pd.Series]) -> pd.DataFrame:
    """多资产收益归因：按资产类别拆分对组合总收益的贡献。

    equity_by_asset: {资产名: 该资产净值序列(等权初始)}
    返回每类资产的总收益与贡献占比。
    """
    if not equity_by_asset:
        return pd.DataFrame(columns=["asset", "total_return", "contribution", "weight"])
    rows = []
    total_final = 0.0
    totals = {}
    for name, eq in equity_by_asset.items():
        s = _guard_equity(eq, require_positive=False)
        _guard_positive_base(s)
        r = float(s.iloc[-1] / s.iloc[0] - 1)
        totals[name] = r
        total_final += float(s.iloc[-1])
    base = sum(float(_guard_equity(eq, require_positive=False).iloc[0])
               for eq in equity_by_asset.values()) or 1.0
    for name, r in totals.items():
        eq = equity_by_asset[name]
        s = _guard_equity(eq, require_positive=False)
        contrib = (float(s.iloc[-1]) - float(s.iloc[0])) / base
        rows.append({
            "asset": name,
            "total_return": round(r, 4),
            "contribution": round(contrib, 4),
            "weight": round(float(s.iloc[-1]) / total_final, 4) if total_final else 0.0,
        })
    return pd.DataFrame(rows)
